#!/usr/bin/env bash
set -e
echo "== Code-RAG MVP Doctor (bash) =="
cmd_ok(){ command -v "$1" >/dev/null 2>&1 && echo "OK" || echo "MISSING"; }
echo "git: $(cmd_ok git)"
echo "python: $(python -V 2>&1 || true)"
echo "uvicorn: $(python -c 'import uvicorn; import sys; print(uvicorn.__version__)' 2>/dev/null || echo MISSING)"
echo "tree_sitter: $(python -c 'import tree_sitter,sys; print(getattr(tree_sitter,\"__version__\",\"unknown\"))' 2>/dev/null || echo MISSING)"
echo "tree_sitter_languages: $(python -c 'import tree_sitter_languages; print(\"OK\")' 2>/dev/null || echo MISSING)"
python - <<'PY' 2>/dev/null || true
from tree_sitter_languages import get_language
from tree_sitter import Query
lang = get_language("javascript")
Query(lang, b"(function_declaration)")
print("Query smoke test: OK")
PY
