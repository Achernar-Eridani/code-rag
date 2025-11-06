#!/usr/bin/env bash
set -euo pipefail

# --------- Config / Defaults ----------
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8081}"

# 日志与运行时目录
LOG_DIR="${LOG_DIR:-./logs}"
RUN_DIR="${RUN_DIR:-./run}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# 载入 .env（可选）
if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source .env
fi

# RAG 环境：默认走本地
export RAG_LLM_PROVIDER="${RAG_LLM_PROVIDER:-local}"
export LOCAL_LLM_BASE="${LOCAL_LLM_BASE:-http://${LLAMA_HOST}:${LLAMA_PORT}/v1}"
# 仅用于 /ping 标签显示（真实 id 由服务端去 /v1/models 读取）
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-qwen2.5-coder-7b-instruct-q6_k}"

LLAMA_LOG="${LOG_DIR}/llama_server.log"
API_LOG="${LOG_DIR}/api_server.log"
LLAMA_PIDFILE="${RUN_DIR}/llama.pid"
API_PIDFILE="${RUN_DIR}/api.pid"

# --------- Helpers ----------
wait_http_ok() {
  local url="$1" name="$2" retry="${3:-60}" sleep_s="${4:-2}"
  for i in $(seq 1 "$retry"); do
    if curl -s "$url" > /dev/null; then
      echo "✓ $name OK at $url"
      return 0
    fi
    echo "  ... ($i/$retry) waiting $name"
    sleep "$sleep_s"
  done
  echo "✗ $name not ready after $((retry*sleep_s))s ($url)"
  return 1
}

# 简易端口占用提示（仅提示，不终止）
check_port_hint() {
  local port="$1" tag="$2"
  if command -v netstat.exe >/dev/null 2>&1; then
    if netstat.exe -ano | tr -d '\r' | awk '{print $2" "$3" "$4" "$5" "$6}' | grep -q ":${port} "; then
      echo "⚠ Port ${port} may be in use before starting ${tag}"
    fi
  fi
}

# --------- 1) Start llama.cpp ----------
echo "== Starting llama.cpp server =="
check_port_hint "${LLAMA_PORT}" "llama.cpp"
# 后台启动；由 day7-start-llama.sh 决定 Docker / Binary
./scripts/day7-start-llama.sh >> "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
echo "$LLAMA_PID" > "$LLAMA_PIDFILE"
echo "  [pid=$LLAMA_PID] logs -> $LLAMA_LOG"

# 健康检查（/health 与 /v1/models 任一即可）
if ! wait_http_ok "http://${LLAMA_HOST}:${LLAMA_PORT}/health" "llama.cpp(/health)" 60 2; then
  echo "查看日志：$LLAMA_LOG"
  exit 1
fi
# 再尝试 /v1/models（增强）
wait_http_ok "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" "llama.cpp(/v1/models)" 10 1 || true

# --------- 2) Start FastAPI ----------
echo "== Starting FastAPI =="
check_port_hint "${API_PORT}" "FastAPI"
python -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" --reload >> "$API_LOG" 2>&1 &
API_PID=$!
echo "$API_PID" > "$API_PIDFILE"
echo "  [pid=$API_PID] logs -> $API_LOG"

# /ping 健康检查（期望 ok:true 且 provider=local）
for i in $(seq 1 60); do
  RESP="$(curl -s "http://${API_HOST}:${API_PORT}/ping" || true)"
  if echo "$RESP" | grep -q '"ok": *true' && echo "$RESP" | grep -q '"provider": *"local"'; then
    echo "✓ FastAPI /ping OK"
    break
  fi
  echo "  ... ($i/60) waiting FastAPI /ping"
  sleep 2
done

echo
echo "== All services up =="
echo "  llama.cpp : http://${LLAMA_HOST}:${LLAMA_PORT}/v1"
echo "  FastAPI   : http://${API_HOST}:${API_PORT}"
echo "  Logs      : $LLAMA_LOG, $API_LOG"
echo "  PID files : $LLAMA_PIDFILE, $API_PIDFILE"

# 前台阻塞，便于 Ctrl+C 触发 trap
trap '
  echo; echo "== Shutting down ==";
  if [ -f "'"$API_PIDFILE"'" ]; then kill $(cat "'"$API_PIDFILE"'") 2>/dev/null || true; rm -f "'"$API_PIDFILE"'"; fi
  if [ -f "'"$LLAMA_PIDFILE"'" ]; then kill $(cat "'"$LLAMA_PIDFILE"'") 2>/dev/null || true; rm -f "'"$LLAMA_PIDFILE"'"; fi
  exit 0
' INT TERM

# 简单地 tail -f 保持前台（可 Ctrl+C 退出触发 trap）
tail -f /dev/null
