# indexer/ast_dump.py 修正版
import argparse, json, sys, pathlib
from typing import Iterable
from tree_sitter import Parser
from tree_sitter_languages import get_language

EXT2LANG = {
    ".js": "javascript",
    ".jsx": "javascript", 
    ".ts": "typescript",
    ".tsx": "tsx",
}
JS_LIKE = set(EXT2LANG.keys())

# 注意：这里用字符串而不是bytes
QUERY_SRC = """
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

def emit(fh, file_rel, kind, name_text, start_line, end_line):
    fh.write(json.dumps({
        "file": file_rel,
        "type": kind,
        "name": name_text,
        "start": start_line,
        "end": end_line,
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
            try:
                # 简化处理：所有JS-like文件都用javascript parser
                # （Day 1够用，Day 2再区分typescript）
                lang = get_language("javascript")
                parser = Parser()
                parser.set_language(lang)
                
                src = file.read_bytes()
                tree = parser.parse(src)
                
                # 使用 language.query() 方法而不是 Query 类
                query = lang.query(QUERY_SRC)
                
                file_rel = str(file.relative_to(repo))
                
                # 使用captures方法（更稳定）
                captures = query.captures(tree.root_node)
                
                seen = set()  # 去重
                for node, capture_name in captures:
                    if capture_name == "name":
                        parent = node.parent
                        name_text = node.text.decode("utf-8", "ignore")
                        start = parent.start_point[0] + 1
                        end = parent.end_point[0] + 1
                        
                        # 简单去重
                        key = (file_rel, name_text, start)
                        if key not in seen:
                            seen.add(key)
                            emit(fh, file_rel, parent.type, name_text, start, end)
                    
                    elif capture_name == "var_name":
                        # 处理箭头函数
                        name_text = node.text.decode("utf-8", "ignore")
                        parent = node.parent
                        start = parent.start_point[0] + 1
                        end = parent.end_point[0] + 1
                        
                        key = (file_rel, name_text, start)
                        if key not in seen:
                            seen.add(key)
                            emit(fh, file_rel, "arrow_function", name_text, start, end)
                            
            except Exception as e:
                print(f"Warning: Failed to parse {file}: {e}", file=sys.stderr)
                continue
    
    print(f"Wrote -> {outp}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[AST] Error: {e}", file=sys.stderr)
        sys.exit(1)