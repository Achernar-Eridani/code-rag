# indexer/chunker.py
"""
Day 2: AST-aware chunking v1 (stable schema, captures-only)
- 用 language.query() + query.captures(root)（不使用 matches）
- 覆盖：
  * function_declaration
  * class_declaration / method_definition
  * variable_declarator = (arrow_function | function)
  * assignment_expression left = (identifier | member_expression),
    right = (arrow_function | function)
- 输出 schema: { "id": str, "text": str, "meta": {...} }
"""
from __future__ import annotations
import argparse, json, os, pathlib, hashlib
from typing import List, Optional
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language

# 不按固定行数切代码，而是按 AST 节点切，默认一个 chunk 最多 200 行，并且会把紧邻上方的注释当作 docstring 拼进 chunk。 ----
CHUNK_MAX_LINES = int(os.getenv("CHUNK_MAX_LINES", "200"))
DOC_MAX_LINES   = int(os.getenv("DOC_MAX_LINES", "20"))
COMMENT_GAP     = int(os.getenv("COMMENT_GAP", "2"))

EXT2LANG = {
    ".js":"javascript", ".jsx":"javascript",
    ".ts":"typescript", ".tsx":"tsx",
    ".mjs":"javascript", ".cjs":"javascript",
}
SUPPORTED_EXTS = set(EXT2LANG.keys())

# --- 新增：过滤规则 ---
# 忽略的目录名（全匹配）
IGNORE_DIRS = {
    "node_modules", "dist", "build", ".git", "coverage", 
    "test", "tests", "__tests__", "spec", "specs", "e2e", 
    ".vscode", ".idea", ".github"
}
# 忽略的文件后缀（endsWith 匹配）
IGNORE_SUFFIXES = {
    ".d.ts", ".spec.ts", ".test.ts", ".spec.js", ".test.js", 
    ".min.js", ".map", ".spec.tsx", ".test.tsx"
}

# 需要哪些“容器节点”作为 chunk 单位 只捕获“容器结点”，名字统一在代码里计算（避免 matches 的配对差异）
QUERY_STR = r"""
; comments for doc
(comment) @comment

(function_declaration) @function
(class_declaration) @class
(method_definition) @method

; const foo = () => {}
(variable_declarator
  value: (arrow_function)) @arrow_var

; const foo = function (...) { ... }
(variable_declarator
  value: (function)) @var_fn

; foo = () => {}   或   foo = function (...) {}
(assignment_expression
  left: (_)
  right: [(arrow_function) (function)]) @assign_stmt
"""

def make_query(lang):
    try:
        return lang.query(QUERY_STR)
    except Exception:
        return lang.query(QUERY_STR.encode("utf-8"))

# ---- utils ----
def repo_rel(root: pathlib.Path, p: pathlib.Path) -> str:
    return str(p.relative_to(root))

def node_text(src: bytes, n: Node) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "ignore")

def limit_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    return text if len(lines) <= max_lines else "\n".join(lines[:max_lines])

# 向上爬 AST，看是否包在 export_statement 里
def is_exported(n: Node) -> bool:
    cur = n
    while cur is not None:
        if cur.type == "export_statement":
            return True
        cur = cur.parent
    return False

# 向上找到最近的 class_declaration，提取 class 名 知道自己父类
def parent_class_name(n: Node, src: bytes) -> Optional[str]:
    cur = n
    while cur is not None:
        if cur.type == "class_declaration":
            nm = cur.child_by_field_name("name")
            if nm:
                return node_text(src, nm)
        cur = cur.parent
    return None

# 沿着父链向上，将 class/function/method 名按“层级”拼成一条路径
def build_ast_path(n: Node, src: bytes) -> str:
    parts = []
    cur = n
    while cur is not None:
        if cur.type == "function_declaration":
            nm = cur.child_by_field_name("name")
            if nm: parts.append(f"function:{node_text(src, nm)}")
        elif cur.type == "class_declaration":
            nm = cur.child_by_field_name("name")
            if nm: parts.append(f"class:{node_text(src, nm)}")
        elif cur.type == "method_definition":
            nm = cur.child_by_field_name("name")
            if nm: parts.append(f"method:{node_text(src, nm)}")
        cur = cur.parent
    parts.reverse()
    return "/".join(parts)

# 压缩多余空白 相当于函数签名/定义摘要 方便embedding模型理解并且便于展示
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

# 收集注释
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

# 拼接注释 作为chunk文档说明
def preceding_doc(n: Node, src: bytes, comments: List[Node]) -> Optional[str]:
    start_line = n.start_point[0]
    picked = []
    for c in comments:
        if c.end_point[0] < start_line and (start_line - c.end_point[0]) <= COMMENT_GAP:
            picked.append(node_text(src, c))
    if not picked:
        return None
    return limit_lines("\n".join(picked), DOC_MAX_LINES)

# 给每个chunk唯一id
def stable_id(path: str, kind: str, name: str, start: int, end: int) -> str:
    h = hashlib.md5(f"{path}|{kind}|{name}|{start}|{end}".encode("utf-8")).hexdigest()
    return h[:12]

# ---- name helpers (不依赖 matches 的配对关系) ----
def name_for_function_like(node: Node, src: bytes) -> Optional[str]:
    """function_declaration / class_declaration / method_definition / variable_declarator"""
    nm = node.child_by_field_name("name")
    return node_text(src, nm) if nm else None

def name_for_assignment_left(left: Node, src: bytes) -> str:
    """assignment_expression.left: identifier | member_expression | others"""
    if left.type == "identifier":
        return node_text(src, left)
    if left.type == "member_expression":
        # 优先取 field: property
        prop = left.child_by_field_name("property")
        if prop:
            return node_text(src, prop)
        # 退化：找 property_identifier
        for i in range(left.named_child_count):
            ch = left.named_child(i)
            if ch.type == "property_identifier":
                return node_text(src, ch)
        # 再退化：整个表达式文本
        return node_text(src, left)
    # 其它形态：整体文本
    return node_text(src, left)

# ---- core ----
# 单个文件切 chunk
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

    # 直接用 captures 扫描所有容器结点
    for node, cap in query.captures(tree.root_node):
        kind = None
        name = None
        target = node  # 默认把捕获到的 node 当作代码容器

        if cap == "function":               # function_declaration
            kind = "function"
            name = name_for_function_like(node, src)

        elif cap == "class":                # class_declaration
            kind = "class"
            name = name_for_function_like(node, src)

        elif cap == "method":               # method_definition
            kind = "method"
            name = name_for_function_like(node, src)

        elif cap == "arrow_var":            # variable_declarator value: arrow_function
            kind = "arrow_function"
            name = name_for_function_like(node, src)

        elif cap == "var_fn":               # variable_declarator value: function
            kind = "function"
            name = name_for_function_like(node, src)

        elif cap == "assign_stmt":          # assignment_expression left: _ ; right: (arrow|function)
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if right is not None:
                if right.type == "arrow_function":
                    kind = "arrow_function"
                elif right.type == "function":
                    kind = "function"
                else:
                    continue
            else:
                continue
            if left is None:
                continue
            name = name_for_assignment_left(left, src)

        else:
            continue

        if not name:
            continue
        
        start = target.start_point[0] + 1
        end   = target.end_point[0] + 1
        dedup_key = (name, start, kind)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        
        sig = extract_signature(target, src)
        doc = preceding_doc(target, src, comments)
        code_full = node_text(src, target)
        code = limit_lines(code_full, CHUNK_MAX_LINES)
        exported = is_exported(target)
        parent_cls = parent_class_name(target, src)
        astpath = build_ast_path(target, src)

        text = "\n".join([p for p in [sig, doc, code] if p])
        meta = {
            "path": rel, "language": lang_name, "kind": kind, "name": name,
            "start_line": start, "end_line": end, "ast_path": astpath,
            "exported": exported, "parent_class": parent_cls, "signature": sig,
            "docstring_len": 0 if not doc else len(doc.splitlines()),
            "code_len": len(code.splitlines()),
        }
        cid = stable_id(rel, kind, name, start, end)
        chunks.append({"id": cid, "text": text, "meta": meta})
    return chunks

def chunk_repo(repo_root: pathlib.Path, out_path: pathlib.Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    src_files = []
    # 使用 rglob 遍历
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        
        # 1. 过滤忽略的目录 (使用 relative_to 确保匹配的是目录部分)
        try:
            rel_parts = p.relative_to(repo_root).parts
        except ValueError:
            rel_parts = p.parts # fallback
            
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
            
        # 2. 过滤忽略的后缀
        if any(p.name.endswith(suf) for suf in IGNORE_SUFFIXES):
            continue

        if p.suffix.lower() in SUPPORTED_EXTS:
            src_files.append(p)
            
    print(f"Found {len(src_files)} source files under {repo_root}")
    
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in src_files:
            recs = chunk_file(repo_root, p)
            for rec in recs:
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
        print(f"⚠ Only {total} (<500). Consider a bigger repo (or your own TS project).")

if __name__ == "__main__":
    main()