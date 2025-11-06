set -euo pipefail

# Try to kill by known pids (if stored) or by port
echo "Killing uvicorn (8000) and llama server (8081) by port..."
# Windows Git Bash 下可用
# uvicorn
pid_api=$(netstat -ano | grep ":8000" | awk '{print $5}' | head -n1)
# llama
pid_llm=$(netstat -ano | grep ":8081" | awk '{print $5}' | head -n1)

[ -n "${pid_api:-}" ] && taskkill //PID "$pid_api" //F || true
[ -n "${pid_llm:-}" ] && taskkill //PID "$pid_llm" //F || true

echo "Done."
