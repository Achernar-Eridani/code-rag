set -euo pipefail

# 按优先级查找模型
find_model() {
    for model in \
        "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
        "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf" \
        "models/*.gguf"
    do
        if [ -f "$model" ]; then
            echo "$model"
            return 0
        fi
    done
    return 1
}

# 自动检测GPU并设置层数
detect_gpu_layers() {
    if nvidia-smi >/dev/null 2>&1; then
        # NVIDIA GPU detected
        echo "35"  # 7B模型的合理默认值
    elif rocm-smi >/dev/null 2>&1; then
        # AMD GPU
        echo "35"
    else
        # No GPU
        echo "0"
    fi
}

# ---- Config ----
MODEL_PATH="${MODEL_PATH:-$(find_model)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-16384}"
# 生成速度 & 质量折中
THREADS="${THREADS:-4}"        # CPU 线程；有 GPU 也需少量 CPU
BATCH="${BATCH:-512}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.95}"
# GPU 图层（如果llama.cpp 编译了 CUDA/ROCM，自动会用；不确定就先不强制）
N_GPU_LAYERS="${N_GPU_LAYERS:-$(detect_gpu_layers)}"   # 0=纯CPU；稳定后可设 >0（如 35）

# ---- Binary path ----
# 需要有 llama.cpp 的 server 可执行文件（例如 ./llama.cpp/server 或可执行发行包）
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./llama.cpp/server}"

if [ -z "$MODEL_PATH" ]; then
    echo "✗ No GGUF model found in models/"
    echo "Please download a model, e.g.:"
    echo "  wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
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
  --ubatch "$UBATCH" \
  --n-gpu-layers "$N_GPU_LAYERS"
