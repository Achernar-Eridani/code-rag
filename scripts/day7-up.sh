set -euo pipefail

# Load env if present
if [ -f ".env" ]; then
  source .env
fi

# Defaults
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8081}"

# 1) Start llama.cpp server (background)
echo "== Starting llama.cpp server =="
# 你可以在另一个终端独立启动；这里默认同终端后台起
./scripts/day7-start-llama.sh &> /tmp/llama_server.log &
LLAMA_PID=$!

# 2) Wait for health
echo "== Waiting for llama.cpp health =="
for i in {1..60}; do
  if curl -s "http://${LLAMA_HOST}:${LLAMA_PORT}/health" > /dev/null; then
    echo "✓ llama.cpp is healthy"
    break
  fi
  echo "  ... ($i) waiting ..."
  sleep 2
done

if ! curl -s "http://${LLAMA_HOST}:${LLAMA_PORT}/health" > /dev/null; then
  echo "✗ llama.cpp server not healthy after timeout. Check /tmp/llama_server.log"
  kill $LLAMA_PID || true
  exit 1
fi

# 3) Start FastAPI
echo "== Starting FastAPI =="
python -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" --reload &> /tmp/api_server.log &
API_PID=$!

echo "== All services up =="
echo "  llama.cpp : http://${LLAMA_HOST}:${LLAMA_PORT}/v1"
echo "  FastAPI   : http://${API_HOST}:${API_PORT}"

# 4) Trap to down
trap 'echo; echo "Shutting down..."; kill $API_PID $LLAMA_PID 2>/dev/null || true; exit 0' INT TERM

# 5) Wait
wait
