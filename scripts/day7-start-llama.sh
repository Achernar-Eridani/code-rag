set -euo pipefail

# ---- Config ----
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-coder-7b-instruct-q4_k_m.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
# 上下文：先 8192 起步；确认稳定后可试 16k/32k（会增内存）
CTX_SIZE="${CTX_SIZE:-8192}"
# 生成速度 & 质量折中
THREADS="${THREADS:-8}"        # CPU 线程；有 GPU 也需少量 CPU
BATCH="${BATCH:-512}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.95}"
# GPU 图层（如果llama.cpp 编译了 CUDA/ROCM，自动会用；不确定就先不强制）
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"   # 0=纯CPU；稳定后可设 >0（如 35）

# ---- Binary path ----
# 需要有 llama.cpp 的 server 可执行文件（例如 ./llama.cpp/server 或可执行发行包）
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./llama.cpp/server}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "✗ Model not found: $MODEL_PATH"
  echo "Please place your GGUF under ./models and set MODEL_PATH if needed."
  exit 1
fi

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "✗ llama.cpp server binary not found or not executable: $LLAMA_SERVER_BIN"
  echo "Please build/download llama.cpp and set LLAMA_SERVER_BIN accordingly."
  exit 1
fi

echo "Starting llama.cpp server..."
echo "  model     : $MODEL_PATH"
echo "  host:port : $HOST:$PORT"
echo "  ctx       : $CTX_SIZE"
echo "  n_gpu_layers: $N_GPU_LAYERS"

# 注意：不同版本参数名略有差异，以下组合在多数发行版可用
"$LLAMA_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --batch "$BATCH" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --n-gpu-layers "$N_GPU_LAYERS"
