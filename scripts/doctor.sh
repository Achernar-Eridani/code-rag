#!/usr/bin/env bash
set -e
echo "== Code-RAG MVP Doctor (bash) =="
python - <<'PY'
import sys, importlib
print("python:", sys.version.split()[0])
def check(name):
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK ({getattr(m,'__version__','unknown')})")
    except Exception as e:
        print(f"{name}: MISSING ({e.__class__.__name__})")
for n in ["uvicorn","tree_sitter","tree_sitter_languages"]:
    check(n)
from tree_sitter_languages import get_language
lang = get_language("javascript")
ok = False
try:
    lang.query("(function_declaration)")
    print("Query smoke: OK (str)")
    ok = True
except Exception as e:
    print("Query(str) failed:", e.__class__.__name__)
    try:
        lang.query(b"(function_declaration)")
        print("Query smoke: OK (bytes)")
        ok = True
    except Exception as e2:
        print("Query(bytes) failed:", e2.__class__.__name__)
print("Query usable:", ok)
PY
