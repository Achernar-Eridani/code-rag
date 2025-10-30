import argparse, json, sys, pathlib
from typing import Iterable
from tree_sitter_languages import get_language, get_parser

JS_LIKE = {".js", ".jsx", ".ts", ".tsx"}

# 三类基本节点：函数 / 类 / 方法（够 Day 1 演示）
QUERY = """
(function_declaration name: (identifier) @name) @fn
(class_declaration name: (identifier) @name) @class
(method_definition name: (property_identifier) @name) @method
(export_statement (function_declaration name: (identifier) @name)) @export_fn
"""

def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in JS_LIKE and p.is_file():
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True, help="Path to JS/TS repo")
    ap.add_argument("--out", type=str, default="data/ast_min.jsonl")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 这里用 JS 解析器（TS/JS/TSX/JSX 都先用它粗略跑通 Day 1）
    lang = get_language("javascript")
    parser = get_parser("javascript")
    query = lang.query(QUERY)

    with outp.open("w", encoding="utf-8") as f:
        for file in iter_files(repo):
            src = file.read_bytes()
            tree = parser.parse(src)
            caps = query.captures(tree.root_node)
            # caps: list[(Node, capture_name)]
            for node, cap in caps:
                if cap == "name":
                    # 只在 name 捕获处输出一次
                    start_line = node.parent.start_point[0] + 1  # 1-based
                    end_line = node.parent.end_point[0] + 1
                    kind = node.parent.type
                    obj = {
                        "file": str(file.relative_to(repo)),
                        "type": kind,
                        "name": node.text.decode("utf-8", errors="ignore"),
                        "start": start_line,
                        "end": end_line,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote -> {outp}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[AST] Error: {e}", file=sys.stderr)
        sys.exit(1)
