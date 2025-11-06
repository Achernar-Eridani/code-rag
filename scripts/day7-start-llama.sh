#!/usr/bin/env bash
set -euo pipefail

# ---------- Mode selection ----------
# 优先 Docker；若设置 LLAMA_USE_BINARY=1 或没有 docker 则用二进制
HAS_DOCKER=0
if command -v docker >/dev/null 2>&1; then
  HAS_DOCKER=1
fi
USE_BINARY="${LLAMA_USE_BINARY:-0}"
if [ "$USE_BINARY" = "1" ] || [ "$HAS_DOCKER" -eq 0 ]; then
  START_MODE="binary"
else
  START_MODE="docker"
fi

# ---------- Common config ----------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"         # 对外暴露端口（Docker 会映射到容器 8080）
CTX_SIZE="${CTX_SIZE:-8192}" # 7B 建议 8k/16k 都可；按需调大

# 自动检测 GPU，决定 n_gpu_layers（35 是 7B 常用）
detect_gpu_layers() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "${N_GPU_LAYERS:-35}"
  elif command -v rocm-smi >/dev/null 2>&1; then
    echo "${N_GPU_LAYERS:-35}"
  else
    echo "0"
  fi
}

# ---------- Model picking ----------
# models 目录优先查找 7B q6_k -> 7B q4_k_m -> 3B q4_k_m -> 任意 .gguf
pick_model_file() {
  local base="${1}"
  local cand
  for cand in \
    "qwen2.5-coder-7b-instruct-q6_k.gguf" \
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
    "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
  do
    if [ -f "${base}/${cand}" ]; then
      echo "${cand}"
      return 0
    fi
  done
  # 兜底：任意一个 .gguf
  local any
  any=$(ls -1 "${base}"/*.gguf 2>/dev/null | head -n1 || true)
  if [ -n "${any:-}" ]; then
    basename "$any"
    return 0
  fi
  return 1
}

# ---------- Start (docker/binary) ----------
if [ "$START_MODE" = "docker" ]; then
  # Windows Git Bash 下的路径转换保护
  UNAME_S="$(uname -s || echo "")"
  if echo "$UNAME_S" | grep -qi "mingw"; then
    if [ "${MSYS_NO_PATHCONV:-}" != "1" ] || [ "${MSYS2_ARG_CONV_EXCL:-}" != "*" ]; then
      echo "✗ On Git Bash, please run:"
      echo "    export MSYS_NO_PATHCONV=1"
      echo '    export MSYS2_ARG_CONV_EXCL="*"'
      exit 1
    fi
  fi

  MODELS_DIR="${MODELS_DIR:-D:/code-rag/models}"  # 你的默认位置
  if [ ! -d "$MODELS_DIR" ]; then
    echo "✗ MODELS_DIR not found: $MODELS_DIR"
    echo "  Set MODELS_DIR or create the directory and put *.gguf there."
    exit 1
  fi

  MODEL_BN="${MODEL_BN:-$(pick_model_file "$MODELS_DIR")}"
  if [ -z "${MODEL_BN:-}" ]; then
    echo "✗ No GGUF model found under $MODELS_DIR"
    exit 1
  fi

  NGL="$(detect_gpu_layers)"
  # 自动选择镜像（有 GPU 就用 cuda 版）
  if command -v nvidia-smi >/dev/null 2>&1; then
    IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
    GPU_FLAGS="--gpus all"
  else
    IMAGE="ghcr.io/ggml-org/llama.cpp:server"
    GPU_FLAGS=""
    NGL="0"
  fi

  echo "Starting llama.cpp (docker)..."
  echo "  image     : $IMAGE"
  echo "  models    : $MODELS_DIR (-> /models)"
  echo "  model     : $MODEL_BN"
  echo "  host:port : $HOST:$PORT"
  echo "  n_gpu_layers: $NGL"

  # 前台运行（由上层脚本放后台并写 PID）
  exec docker run --rm $GPU_FLAGS \
    -p "${PORT}:8080" \
    -v "${MODELS_DIR}:/models:ro" \
    "$IMAGE" \
    -m "/models/${MODEL_BN}" \
    -ngl "$NGL" -c "$CTX_SIZE"

else
  # 二进制模式
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./llama.cpp/server}"
  if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "✗ llama.cpp server binary not found or not executable: $LLAMA_SERVER_BIN"
    echo "  Set LLAMA_SERVER_BIN or use docker by unsetting LLAMA_USE_BINARY."
    exit 1
  fi

  MODELS_DIR="${MODELS_DIR:-./models}"
  if [ ! -d "$MODELS_DIR" ]; then
    echo "✗ models dir not found: $MODELS_DIR"
    exit 1
  fi
  MODEL_BN="${MODEL_BN:-$(pick_model_file "$MODELS_DIR")}"
  if [ -z "${MODEL_BN:-}" ]; then
    echo "✗ No GGUF model found under $MODELS_DIR"
    exit 1
  fi

  THREADS="${THREADS:-4}"
  BATCH="${BATCH:-512}"
  UBATCH="${UBATCH:-256}"
  TOP_K="${TOP_K:-40}"
  TOP_P="${TOP_P:-0.95}"
  NGL="$(detect_gpu_layers)"

  echo "Starting llama.cpp (binary)..."
  echo "  bin       : $LLAMA_SERVER_BIN"
  echo "  model     : $MODELS_DIR/$MODEL_BN"
  echo "  host:port : $HOST:$PORT"
  echo "  ctx       : $CTX_SIZE"
  echo "  threads   : $THREADS  batch:$BATCH ubatch:$UBATCH"
  echo "  n_gpu_layers: $NGL"

  exec "$LLAMA_SERVER_BIN" \
    -m "${MODELS_DIR}/${MODEL_BN}" \
    --host "$HOST" --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --batch "$BATCH" \
    --ubatch "$UBATCH" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --n-gpu-layers "$NGL"
fi
