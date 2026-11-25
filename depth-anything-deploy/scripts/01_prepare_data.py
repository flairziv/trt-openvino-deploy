"""准备评测集与校准集,生成 data/nyu/manifest.json。

输入(data/raw/,由 tools/ 下的下载脚本获得):
    nyu_val-000000.tar / nyu_val-000001.tar   654 张官方 test 划分(h5)
    splits.mat                                官方划分索引(testNdxs,1-based)
    train_window_*.bin                        12 个训练 tar 的中段 40 MB 字节窗口

输出:
    data/nyu/test/*.h5     654 张测试图
    data/nyu/calib/*.h5    calib 200 张 + val 100 张(训练场景,与 test 不相交)
    data/nyu/manifest.json 划分清单 + 来源与校验信息

关键步骤:
    1. 解包 test,并用 splits.mat 的 testNdxs 逐个核对文件名 —— 证明评测集就是官方 654 张;
    2. 从字节窗口里"扫描 tar 头"恢复完整成员(窗口两端被截断的成员丢弃);
    3. 固定种子随机抽样 calib / val 子集,两者互斥。

用法:
    python scripts/01_prepare_data.py
"""

from __future__ import annotations

import json
import random
import sys
import tarfile
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dadeploy.config import (  # noqa: E402
    CALIB_NUM,
    CALIB_SEED,
    MANIFEST_PATH,
    NYU_CALIB_DIR,
    NYU_TEST_DIR,
    RAW_DIR,
    VAL_NUM,
)
from dadeploy.logging_ import get_logger  # noqa: E402

# 模块级 logger
log = get_logger("prepare")

# tar 的块大小(字节)
_TAR_BLOCK = 512
# 官方 test 划分的图片数
_EXPECTED_TEST = 654


def extract_test() -> list[str]:
    """解包两个 val tar 到 data/nyu/test/,返回相对清单路径(按索引排序)。"""
    # 创建测试集目录
    NYU_TEST_DIR.mkdir(parents=True, exist_ok=True)
    # 按文件名顺序处理每个 val tar 包
    for tar_path in sorted(RAW_DIR.glob("nyu_val-*.tar")):
        # 打开 tar 包
        with tarfile.open(tar_path) as tf:
            # 遍历包内成员
            for member in tf.getmembers():
                # 只处理 h5 普通文件
                if not member.isfile() or not member.name.endswith(".h5"):
                    continue
                # 目标路径只保留文件名
                dst = NYU_TEST_DIR / Path(member.name).name
                # 已存在的文件不重复写
                if not dst.exists():
                    dst.write_bytes(tf.extractfile(member).read())
    # 列出解包后的全部 h5 文件名并排序
    names = sorted(p.name for p in NYU_TEST_DIR.glob("*.h5"))
    # 数量必须等于官方的 654 张
    if len(names) != _EXPECTED_TEST:
        raise RuntimeError(f"expect {_EXPECTED_TEST} test images, got {len(names)}")
    # 返回相对数据目录的路径
    return [f"test/{n}" for n in names]


def verify_against_official_split(test_names: list[str]) -> bool:
    """用官方 splits.mat 的 testNdxs 核对测试集文件名集合。

    Returns:
        True 表示与官方 654 张 test 索引完全一致。
    """
    # scipy 只在这里导入
    from scipy.io import loadmat

    # 官方 test 索引的集合
    official = {int(i) for i in loadmat(RAW_DIR / "splits.mat")["testNdxs"].ravel()}
    # 本地测试集文件名对应的索引集合
    ours = {int(Path(n).stem) for n in test_names}
    # 两个集合完全相同才算通过
    ok = official == ours
    # 打印核对结果
    log.info("official split check: %s (official=%d ours=%d)", "PASS" if ok else "FAIL", len(official), len(ours))
    return ok


def scan_tar_window(blob: bytes) -> list[tuple[str, bytes]]:
    """从 tar 文件中段的任意字节窗口里恢复完整成员。

    tar 的每个成员头都对齐在 512 字节边界,头部 257 偏移处有 "ustar" 魔数且带校验和。
    逐块扫描:能被 TarInfo.frombuf 成功解析(校验和通过)且数据完整落在窗口内的,才算有效成员。

    Args:
        blob: 窗口字节(起点已对齐到 512)。

    Returns:
        [(成员名, 文件内容)],只含 .h5。
    """
    # 恢复出的成员列表
    members: list[tuple[str, bytes]] = []
    # 当前扫描位置
    pos = 0
    # 按 512 字节一块向后扫描
    while pos + _TAR_BLOCK <= len(blob):
        # 取一个块作为候选成员头
        header = blob[pos : pos + _TAR_BLOCK]
        # 没有 ustar 魔数的不是成员头,前进一块
        if header[257:262] != b"ustar":
            pos += _TAR_BLOCK
            continue
        # 按 tar 头解析,校验和不通过的不是成员头
        try:
            info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
        except tarfile.HeaderError:
            pos += _TAR_BLOCK
            continue
        # 数据区紧跟在头之后
        data_start = pos + _TAR_BLOCK
        # 数据区的结束位置
        data_end = data_start + info.size
        # 数据超出窗口说明成员被截断,停止扫描
        if data_end > len(blob):
            break  # 最后一个成员被窗口截断
        # 只收集 h5 普通文件
        if info.isfile() and info.name.endswith(".h5"):
            members.append((info.name, blob[data_start:data_end]))
        # 跳到下一个成员头,数据区按 512 字节向上取整
        pos = data_start + ((info.size + _TAR_BLOCK - 1) // _TAR_BLOCK) * _TAR_BLOCK
    return members


def extract_calib_pool() -> list[str]:
    """把所有训练窗口里的完整 h5 写到 data/nyu/calib/,返回相对路径列表。"""
    # 创建校准池目录
    NYU_CALIB_DIR.mkdir(parents=True, exist_ok=True)
    # 写出文件的相对路径列表
    out: list[str] = []
    # 列出全部训练窗口文件
    windows = sorted(RAW_DIR.glob("train_window_*.bin"))
    # 没有窗口文件时提示先下载
    if not windows:
        raise FileNotFoundError("data/raw/train_window_*.bin 不存在,请先运行 tools/fetch_calib_windows.sh")
    # 逐个窗口恢复成员
    for win in windows:
        # 扫描窗口得到完整的 h5 成员
        members = scan_tar_window(win.read_bytes())
        # 这个窗口涉及的场景名
        scenes = sorted({m[0].split("/")[1] for m in members})
        # 打印窗口的帧数与场景
        log.info("%s: %d frames from scenes %s", win.name, len(members), scenes)
        # 逐个成员落盘
        for name, data in members:
            # 成员名拆成前缀、场景名、帧文件名
            _, scene, frame = name.split("/")
            # 落盘文件名:场景名与帧文件名用双下划线连接
            dst_name = f"{scene}__{frame}"
            dst = NYU_CALIB_DIR / dst_name
            # 已存在的文件不重复写
            if not dst.exists():
                dst.write_bytes(data)
            # 记录相对路径
            out.append(f"calib/{dst_name}")
    # 去重并排序后返回
    return sorted(set(out))


# 入口:解包测试集、核对官方划分、抽样校准集与验证集、写清单
def main() -> int:
    # 解包测试集
    test = extract_test()
    # 与官方划分核对
    verified = verify_against_official_split(test)
    # 不一致时终止
    if not verified:
        log.error("测试集与官方划分不一致,终止")
        return 1

    # 恢复训练场景的候选帧
    pool = extract_calib_pool()
    # 校准集与验证集一共需要的张数
    need = CALIB_NUM + VAL_NUM
    # 候选帧不够时报错
    if len(pool) < need:
        raise RuntimeError(f"calib pool too small: {len(pool)} < {need}")
    # 固定种子的随机数发生器
    rng = random.Random(CALIB_SEED)
    # 复制一份再打乱,不改动原列表
    shuffled = pool[:]
    rng.shuffle(shuffled)
    # 前 CALIB_NUM 张作校准集,紧接着的 VAL_NUM 张作验证集
    calib, val = shuffled[:CALIB_NUM], shuffled[CALIB_NUM:need]

    # 清单内容:三个划分和来源信息
    manifest = {
        "test": test,
        "calib": calib,
        "val": val,
        "meta": {
            "test_source": "hf:sayakpaul/nyu_depth_v2 val/official (fast-depth h5)",
            "test_verified_against_splits_mat": verified,
            "calib_source": "hf:sayakpaul/nyu_depth_v2 train-0000xx.tar mid-file 40MB windows",
            "calib_pool_size": len(pool),
            "calib_scenes": sorted({Path(p).name.split("__")[0] for p in calib}),
            "seed": CALIB_SEED,
        },
    }
    # 写出清单
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # 打印各划分的张数
    log.info("manifest: test=%d calib=%d val=%d calib_scenes=%d -> %s",
             len(test), len(calib), len(val), len(manifest["meta"]["calib_scenes"]), MANIFEST_PATH)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
