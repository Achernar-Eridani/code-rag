import argparse, json, sys, pathlib
from typing import Iterable
from tree_sitter import Parser, Query
from tree_sitter_languages import get_language

EXT2LANG = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}
JS_LIKE = set(EXT2LANG.keys())

QUERY_SRC = b"""
(function_declaration name: (identifier) @name) @fn
(class_declaration name: (identifier) @name) @class
(method_definition name: (property_identifier) @name) @method
(export_statement (function_declaration name: (identifier) @name)) @export_fn
(export_statement (class_declaration name: (identifier) @name)) @export_class
(variable_declarator name: (identifier) @var_name value: (arrow_function) @arrow_body) @arrow_var
"""

def iter_files(root: pathlib.Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in JS_LIKE:
            yield p

def emit(fh, file_rel, kind, name_node, container_node):
    fh.write(json.dumps({
        "file": file_rel,
        "type": kind,
        "name": name_node.text.decode("utf-8", "ignore"),
        "start": container_node.start_point[0] + 1,
        "end": container_node.end_point[0] + 1,
    }, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="data/ast_min.jsonl")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as fh:
        for file in iter_files(repo):
            lang_name = EXT2LANG[file.suffix.lower()]
            lang = get_language(lang_name)
            parser = Parser()
            parser.set_language(lang)

            src = file.read_bytes()
            tree = parser.parse(src)
            query = Query(lang, QUERY_SRC)

            # 优先 matches（可取容器结点）
            try:
                for m in query.matches(tree.root_node):
                    caps = {name: node for node, name in m.captures}
                    file_rel = str(file.relative_to(repo))
                    if "name" in caps:
                        container = caps.get("fn") or caps.get("class") or caps.get("method") \
                                   or caps.get("export_fn") or caps.get("export_class")
                        kind = container.type if container else caps["name"].parent.type
                        emit(fh, file_rel, kind, caps["name"], container or caps["name"].parent)
                    elif "var_name" in caps and "arrow_var" in caps:
                        emit(fh, file_rel, "arrow_function", caps["var_name"], caps["arrow_var"])
            except Exception:
                # 兜底：captures 也打一次，保证 Day-1 可用
                for node, cap in query.captures(tree.root_node):
                    if cap == "name":
                        container = node.parent
                        emit(fh, str(file.relative_to(repo)), container.type, node, container)

    print(f"Wrote ->", outp)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[AST] Error:", e, file=sys.stderr)
        sys.exit(1)
