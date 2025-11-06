#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-./run}"
API_PIDFILE="${RUN_DIR}/api.pid"
LLAMA_PIDFILE="${RUN_DIR}/llama.pid"

echo "== Stopping services =="

stop_by_pidfile() {
  local file="$1" name="$2"
  if [ -f "$file" ]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [ -n "${pid:-}" ]; then
      echo "  Killing $name [pid=$pid]"
      kill "$pid" 2>/dev/null || true
      # 等 5s，若仍在则强杀（Windows Git Bash 下 kill -9 同样可用）
      for i in $(seq 1 5); do
        if ps -p "$pid" >/dev/null 2>&1; then
          sleep 1
        else
          break
        fi
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

stop_by_pidfile "$API_PIDFILE" "FastAPI"
stop_by_pidfile "$LLAMA_PIDFILE" "llama.cpp"

# 兜底：Windows 下检查可能残留 docker 容器（名字无法唯一，这里仅提示）
if command -v docker >/dev/null 2>&1; then
  echo "  (hint) If llama.cpp was started via docker and still running, you may run:"
  echo "         docker ps | grep llama.cpp && read -p 'Stop container ID? ' cid && docker stop \"$cid\""
fi

echo "✓ Services stopped."
