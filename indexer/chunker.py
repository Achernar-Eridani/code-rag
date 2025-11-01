# indexer/chunker.py
"""
Day 2: AST-aware chunking v1 (stable schema, wide JS coverage)
- Query backend: language.query()  (str -> bytes fallback)
- Output schema (stable): { "id": str, "text": str, "meta": { ... } }
- Coverage:
  * function_declaration
  * class_declaration / method_definition
  * variable_declarator -> (arrow_function | function | function_expression)
  * assignment_expression -> (identifier | member_expression) = (arrow | function | function_expression)
"""
from __future__ import annotations
import argparse, json, os, pathlib, hashlib
from typing import List, Optional
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language

# ---- config (env) ----
CHUNK_MAX_LINES = int(os.getenv("CHUNK_MAX_LINES", "200"))
DOC_MAX_LINES   = int(os.getenv("DOC_MAX_LINES", "20"))
COMMENT_GAP     = int(os.getenv("COMMENT_GAP", "2"))

EXT2LANG = {
    ".js":"javascript",".jsx":"javascript",
    ".ts":"typescript",".tsx":"tsx",
    ".mjs":"javascript",".cjs":"javascript",
}
SUPPORTED_EXTS = set(EXT2LANG.keys())


QUERY_STR = r"""
; comments (for doc extraction)
(comment) @comment

(function_declaration
  name: (identifier) @fn_name) @function

(class_declaration
  name: (identifier) @class_name
  body: (class_body) @class_body) @class

(method_definition
  name: (property_identifier) @method_name) @method

; const foo = () => {}
(variable_declarator
  name: (identifier) @var_name
  value: (arrow_function) @arrow_body) @arrow_var

; const foo = function (...) { ... }
(variable_declarator
  name: (identifier) @var_name2
  value: (function) @fn_expr) @var_fn

; foo = () => {}
(assignment_expression
  left: (identifier) @assign_name
  right: (arrow_function) @assign_arrow) @assign_arrow_stmt

; foo = function (...) { ... }
(assignment_expression
  left: (identifier) @assign_name2
  right: (function) @assign_fn) @assign_fn_stmt

; obj.prop = function (...) { ... }
(assignment_expression
  left: (member_expression
          object: (_)
          property: (property_identifier) @prop_name)
  right: (function) @assign_prop_fn) @assign_prop_fn_stmt
"""


def make_query(lang):
    try:
        return lang.query(QUERY_STR)
    except Exception as e1:
        try:
            return lang.query(QUERY_STR.encode("utf-8"))
        except Exception as e2:
            raise RuntimeError(f"Query compile failed for language={lang}: {e2 or e1}")


# ---- utils ----
def repo_rel(root: pathlib.Path, p: pathlib.Path) -> str:
    return str(p.relative_to(root))

def node_text(src: bytes, n: Node) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "ignore")

def limit_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines])

def is_exported(n: Node) -> bool:
    cur = n
    while cur is not None:
        if cur.type == "export_statement":
            return True
        cur = cur.parent
    return False

def parent_class_name(n: Node, src: bytes) -> Optional[str]:
    cur = n
    while cur is not None:
        if cur.type == "class_declaration":
            name_node = cur.child_by_field_name("name")
            if name_node:
                return node_text(src, name_node)
        cur = cur.parent
    return None

def build_ast_path(n: Node, src: bytes) -> str:
    parts = []
    cur = n
    while cur is not None:
        if cur.type == "function_declaration":
            name = cur.child_by_field_name("name")
            if name: parts.append(f"function:{node_text(src, name)}")
        elif cur.type == "class_declaration":
            name = cur.child_by_field_name("name")
            if name: parts.append(f"class:{node_text(src, name)}")
        elif cur.type == "method_definition":
            name = cur.child_by_field_name("name")
            if name: parts.append(f"method:{node_text(src, name)}")
        cur = cur.parent
    parts.reverse()
    return "/".join(parts)

def extract_signature(n: Node, src: bytes) -> str:
    raw = node_text(src, n)
    lines = raw.splitlines()
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        if "=>" in line or "{" in line:
            break
    sig = " ".join(" ".join(sig_lines).split())
    return sig[:200]

def collect_comments(root: Node) -> List[Node]:
    out: List[Node] = []
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur.type == "comment":
            out.append(cur)
        for i in range(cur.named_child_count):
            stack.append(cur.named_child(i))
    return out

def preceding_doc(n: Node, src: bytes, comments: List[Node]) -> Optional[str]:
    start_line = n.start_point[0]
    picked: List[str] = []
    for c in comments:
        c_end_line = c.end_point[0]
        if c_end_line < start_line and (start_line - c_end_line) <= COMMENT_GAP:
            picked.append(node_text(src, c))
    if not picked:
        return None
    text = "\n".join(picked)
    return limit_lines(text, DOC_MAX_LINES)

def stable_id(path: str, kind: str, name: str, start: int, end: int) -> str:
    h = hashlib.md5(f"{path}|{kind}|{name}|{start}|{end}".encode("utf-8")).hexdigest()
    return h[:12]

def iter_match_captures(query, match):
    """
    Normalize to (node, name_str):
    - match.captures OR (pattern_index, captures)
    - capture item can be (node, name) / (name, node) / (node, index) / 3-tuples
    """
    caps = getattr(match, "captures", None)
    if caps is None:
        if isinstance(match, tuple) and len(match) >= 2:
            caps = match[1]
        else:
            caps = []

    for item in caps:
        node = None
        label = None
        if isinstance(item, tuple):
            a = item[0] if len(item) > 0 else None
            b = item[1] if len(item) > 1 else None
            if hasattr(a, "type"):
                node, label = a, b
            elif hasattr(b, "type"):
                node, label = b, a
            else:
                continue
        else:
            continue

        if isinstance(label, int):
            try:
                name = query.capture_names[label]
            except Exception:
                name = str(label)
        else:
            name = label if isinstance(label, str) else str(label)

        if node is not None and isinstance(name, str):
            yield node, name

# ---- core ----
def chunk_file(repo_root: pathlib.Path, file_path: pathlib.Path) -> List[dict]:
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return []
    lang_name = EXT2LANG[file_path.suffix.lower()]
    lang = get_language(lang_name)
    parser = Parser(); parser.set_language(lang)

    src = file_path.read_bytes()
    tree = parser.parse(src)
    query = make_query(lang)

    rel = repo_rel(repo_root, file_path)
    comments = collect_comments(tree.root_node)

    chunks: List[dict] = []
    seen_keys = set()

    for match in query.matches(tree.root_node):
        caps = {}
        for node, name in iter_match_captures(query, match):
            if name in {
                "function","class","method","fn_name","class_name","method_name",
                "arrow_var","var_name","arrow_body",
                "var_fn","var_name2","fn_expr",
                "assign_fn_stmt","assign_fn","assign_name2",
                "assign_arrow_stmt","assign_arrow","assign_name",
                "assign_prop_fn_stmt","assign_prop_fn","prop_name"
            }:
                caps[name] = node

        if not caps:
            continue

        # Normalize to (node, name, kind)
        n = None; nm = None; kind = None

        # function declaration
        if "function" in caps and "fn_name" in caps:
            n = caps["function"]; nm = node_text(src, caps["fn_name"]); kind = "function"

        # class declaration
        elif "class" in caps and "class_name" in caps:
            n = caps["class"]; nm = node_text(src, caps["class_name"]); kind = "class"

        # method definition
        elif "method" in caps and "method_name" in caps:
            n = caps["method"]; nm = node_text(src, caps["method_name"]); kind = "method"

        # const foo = () => {}
        elif "arrow_var" in caps and "var_name" in caps:
            n = caps["arrow_var"]; nm = node_text(src, caps["var_name"]); kind = "arrow_function"

        # const foo = function (...) {}
        elif "var_fn" in caps and "var_name2" in caps:
            n = caps["var_fn"]; nm = node_text(src, caps["var_name2"]); kind = "function"

        # foo = () => {}
        elif "assign_arrow_stmt" in caps and "assign_name" in caps:
            n = caps["assign_arrow_stmt"]; nm = node_text(src, caps["assign_name"]); kind = "arrow_function"

        # foo = function (...) {}
        elif "assign_fn_stmt" in caps and "assign_name2" in caps:
            n = caps["assign_fn_stmt"]; nm = node_text(src, caps["assign_name2"]); kind = "function"

        # obj.prop = function (...) {}
        elif "assign_prop_fn_stmt" in caps and "prop_name" in caps:
            n = caps["assign_prop_fn_stmt"]; nm = node_text(src, caps["prop_name"]); kind = "method"

        else:
            continue

        start = n.start_point[0] + 1
        end   = n.end_point[0] + 1
        key = (nm, start, kind)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        sig = extract_signature(n, src)
        doc = preceding_doc(n, src, comments)
        code_full = node_text(src, n)
        code = limit_lines(code_full, CHUNK_MAX_LINES)
        exported = is_exported(n)
        parent_cls = parent_class_name(n, src)
        astpath = build_ast_path(n, src)

        text = "\n".join([p for p in [sig, doc, code] if p])

        meta = {
            "path": rel, "language": lang_name, "kind": kind, "name": nm,
            "start_line": start, "end_line": end, "ast_path": astpath,
            "exported": exported, "parent_class": parent_cls, "signature": sig,
            "docstring_len": 0 if not doc else len(doc.splitlines()),
            "code_len": len(code.splitlines()),
        }
        cid = stable_id(rel, kind, nm, start, end)
        chunks.append({"id": cid, "text": text, "meta": meta})

    return chunks

def chunk_repo(repo_root: pathlib.Path, out_path: pathlib.Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_files = [p for p in repo_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    print(f"Found {len(src_files)} source files under {repo_root}")
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in src_files:
            for rec in chunk_file(repo_root, p):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to JS/TS repo")
    ap.add_argument("--output", default="data/chunks_day2.jsonl", help="output JSONL")
    args = ap.parse_args()

    root = pathlib.Path(args.repo).resolve()
    out  = pathlib.Path(args.output)
    total = chunk_repo(root, out)
    print(f"✓ Generated {total} chunks -> {out}")
    if total < 500:
        print(f"⚠ Only {total} (<500). Consider a larger repo or choose a bigger one.")

if __name__ == "__main__":
    main()
