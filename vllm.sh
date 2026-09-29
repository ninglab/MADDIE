#!/bin/bash
#SBATCH --account=YOUR_SLURM_ACCOUNT
# tests/run_full.sh finds the server by this job name; keep it.
#SBATCH --job-name=vllm
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.error
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2

PORT=8000
NODE=$(hostname)

echo "============================================================"
echo "VLLM is starting on ${NODE}:${PORT}"
echo ""
echo "To access it, run this SSH tunnel on your LOCAL machine:"
echo "  ssh -N -L ${PORT}:${NODE}:${PORT} YOUR_USER@YOUR_CLUSTER"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "============================================================"

export VLLM_LOGGING_LEVEL=DEBUG
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export VLLM_ENGINE_READY_TIMEOUT_S=600
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PARSIZE=2

export OMP_NUM_THREADS=4

export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export NVCC_PREPEND_FLAGS="-ccbin $(which g++)"

module load cuda/13.2.1   # cluster-specific; adjust or drop

uv run --with vllm vllm serve "${QWEN_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}" \
    --download-dir "${HF_HOME:-$HOME/.cache/huggingface}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size $PARSIZE \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
    # --enforce-eager