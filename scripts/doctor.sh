#!/usr/bin/env bash
#!/usr/bin/env bash
set -e
echo "== Code-RAG MVP Doctor (bash) =="
cmd_ok(){ command -v "$1" >/dev/null 2>&1 && echo "OK" || echo "MISSING"; }

echo "git: $(cmd_ok git)"
python - <<'PY'
import sys, importlib, json
print("python:", sys.version.split()[0])
def mod(name):
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK ({getattr(m,'__version__','unknown')})")
    except Exception as e:
        print(f"{name}: MISSING ({e.__class__.__name__})")
for name in ["uvicorn","tree_sitter","tree_sitter_languages"]:
    mod(name)
from tree_sitter_languages import get_language
lang = get_language("javascript")
ok = False
try:
    from tree_sitter import Query
    Query(lang, b"(function_declaration)")
    print("Query smoke: OK (Query(lang, bytes))")
    ok = True
except Exception as e:
    print("Query smoke (Query(lang, bytes)):", e.__class__.__name__)
try:
    lang.query(b"(function_declaration)")
    print("Query smoke: OK (lang.query(bytes))")
    ok = True
except Exception as e:
    print("Query smoke (lang.query(bytes)):", e.__class__.__name__)
print("Query usable:", ok)
PY
