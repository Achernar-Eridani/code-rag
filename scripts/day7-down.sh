#!/usr/bin/env bash
set -euo pipefail

echo "Stopping services..."

# 使用tasklist和taskkill
if command -v tasklist.exe >/dev/null 2>&1; then    
    tasklist.exe | grep -i "server" | awk '{print $2}' | while read pid; do
        taskkill.exe //PID "$pid" //F 2>/dev/null || true
    done
else
    pkill -f "llama.*server" || true
fi

echo "Services stopped."