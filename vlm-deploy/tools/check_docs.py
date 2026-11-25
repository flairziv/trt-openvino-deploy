"""核对 docs/code-walkthrough.md 里的行号有没有过期(只依赖标准库)。

用法:
    python3 tools/check_docs.py            # 有不一致时逐条列出,退出码 1
    python3 tools/check_docs.py --coverage # 另外打印每个文件被导读覆盖了多少行

带行号的文档最容易过期:代码挪一行,后面的行号全错,而读的人不会发现。所以导读按固定格式写,让机器能核对:

    ### `vlmdeploy/config.py`                    ← 标题里反引号包住的仓库相对路径 = 当前文件
    | 行 | 代码 | 说明 |
    |---|---|---|
    | 43–45 | `PATCH_SIZE` `TOKEN_PIXELS` | …… |   ← 第 1 格是行号或"起–止",第 2 格里每个反引号片段都是锚点

规则:每个锚点必须原样出现在该文件的这段行号范围内(子串匹配)。锚点里不能有竖线和反引号(会破坏表格)。
改了代码之后跑一遍,报错的行就是要改的导读行。

能查出什么、查不出什么:代码增删行导致锚点滑出标注范围,能查出;锚点仍在范围内的小幅漂移(范围很宽、锚点很短时)
查不出;第 3 格的文字说明对不对,完全不查 —— 那要靠人读。它是"行号没有明显过期"的检查,不是正确性证明。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 仓库根目录
ROOT = Path(__file__).resolve().parent.parent
# 默认核对的导读文件
DOC = ROOT / "docs" / "code-walkthrough.md"
# 二到五级标题里的第一个反引号片段:可能是文件路径,也可能不是(由 check 判断)
HEADING = re.compile(r"^#{2,5} .*?`([^`]+)`")
# 表格行:第 1 格是行号或"起–止"(连接号写 – 或 - 都认),第 2 格整格取出来找锚点
ROW = re.compile(r"^\| *(\d+)(?:[–-](\d+))? *\|([^|]*)\|")
# 第 2 格里的每个反引号片段
ANCHOR = re.compile(r"`([^`]+)`")
# 标题里的名字带这些后缀就算"看起来是源文件":找不到时要报错,而不是当成普通标题放过
SOURCE_SUFFIXES = {".py", ".sh", ".toml"}


def check(doc: Path) -> tuple[list[str], dict[str, set[int]]]:
    """核对一份导读。

    Returns:
        (错误列表, 每个文件被表格行覆盖到的行号集合)。
    """
    # 错误信息列表
    errors: list[str] = []
    # 每个文件被覆盖到的行号
    covered: dict[str, set[int]] = {}
    current: str | None = None   # 当前小节讲的文件;为 None 时后面的表格行不核对
    source: list[str] = []       # 它的源码,按行
    # 逐行扫描导读,n 是从 1 开始的行号
    for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        # 先看是不是标题行
        heading = HEADING.match(line)
        if heading:
            # 标题里的第一个反引号片段
            name = heading.group(1)
            # 按仓库相对路径解析
            path = ROOT / name
            # 标题里的反引号不一定是路径(也可能是函数名、目录名);只有确实存在的文件才切换"当前文件"
            if path.is_file():
                # 切换当前文件并读入源码
                current = name
                source = path.read_text(encoding="utf-8").splitlines()
                # 为这个文件建立覆盖记录
                covered.setdefault(current, set())
            elif Path(name).suffix in SOURCE_SUFFIXES:
                # 看起来是源文件却不存在(改名、删除):报错并清掉当前文件,免得后面的行被拿去和上一个文件比
                errors.append(f"{doc.name}:{n}: heading names {name}, which does not exist")
                current = None
            continue
        # 再看是不是带行号的表格行
        row = ROW.match(line)
        # 不是表格行或没有当前文件时跳过
        if not row or current is None:
            continue
        # 起始行号
        start = int(row.group(1))
        end = int(row.group(2) or start)   # 只写一个行号时,止 = 起
        # 行号超出文件范围时报错
        if not 1 <= start <= end <= len(source):
            errors.append(f"{doc.name}:{n}: {current} has {len(source)} lines, row says {start}–{end}")
            continue
        # 记录覆盖到的行号
        covered[current].update(range(start, end + 1))
        # 这段行号范围内的源码
        block = source[start - 1:end]
        # 逐个锚点核对
        for anchor in ANCHOR.findall(row.group(3)):
            # 逐行做子串匹配,所以锚点不能跨行
            if not any(anchor in text for text in block):
                errors.append(f"{doc.name}:{n}: `{anchor}` not found in {current}:{start}–{end}")
    return errors, covered


def main() -> int:
    """命令行入口:核对导读的行号,可选打印覆盖率。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="核对代码导读里的行号")
    # 要核对的导读文件
    parser.add_argument("--doc", type=Path, default=DOC)
    # 打印覆盖率
    parser.add_argument("--coverage", action="store_true", help="打印每个文件被导读覆盖的行数")
    # 解析参数
    args = parser.parse_args()

    # 核对导读
    errors, covered = check(args.doc)
    # 逐个文件打印覆盖的行数和总行数
    if args.coverage:
        for path, lines in covered.items():
            total = len((ROOT / path).read_text(encoding="utf-8").splitlines())
            print(f"{path:44s} {len(lines):4d}/{total:4d} lines")
    # 错误信息写到标准错误
    for err in errors:
        print(err, file=sys.stderr)
    # 带行号的表格行总数
    rows = sum(1 for line in args.doc.read_text(encoding="utf-8").splitlines() if ROW.match(line))
    # 打印文件数、表格行数和不一致的条数
    print(f"{args.doc.name}: {len(covered)} files, {rows} rows, {len(errors)} mismatches")
    # 有不一致时退出码为 1
    return 1 if errors else 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
