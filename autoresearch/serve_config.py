# serve_config.py — Baseline serving configuration for DGX Spark / GB10 (sm_121)
#
# This file is the single source of truth for default serving parameters.
# Each queue.yaml entry overrides specific values via its `config:` block.
# Do NOT change this file mid-experiment; instead queue a new experiment that
# overrides the relevant field.
#
# Hardware: 2x NVIDIA GB10 Grace Blackwell Superchip (SM121, arm64, 128GB unified memory)
# Topology: Pipeline Parallel PP=2 across spark-01 + spark-02 via K8s + Ray
# Inter-node: 10GbE (no NVLink between nodes)

FRAMEWORK = "vllm"
MODEL = "Qwen/Qwen3-Coder-Next-FP8"

# Container images
VLLM_IMAGE = "ghcr.io/elizabetht/inference-images/vllm-serve:0.0.1"
SGLANG_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20260318-cb1e63ab"

# Quantization / dtype
DTYPE = "bfloat16"             # Qwen3-Coder-Next-FP8 has pre-quantized FP8 weights
KV_CACHE_DTYPE = "auto"        # "fp8" for FP8 KV cache (not yet validated on GB10)

# Memory — conservative for PP=2 Ray overhead on 128GB unified memory
GPU_MEMORY_UTILIZATION = 0.75  # leave headroom; Ray OOM kills at 0.85+
MAX_MODEL_LEN = 131072         # 128K; Qwen3-Coder-Next supports 1M but 2x128GB limits

# Parallelism — PP=2 is the proven cross-node topology on 10GbE
PP_SIZE = 2
TP_SIZE = 1                    # TP=1 within each node; TP>1 on Blackwell needs --disable-custom-all-reduce

# Sequence batching
MAX_NUM_SEQS = 256

# Attention
ENABLE_PREFIX_CACHING = True   # APC default-on; good for code workloads
ENABLE_CHUNKED_PREFILL = False  # off by default; chunked-prefill-pp2 experiment
ENFORCE_EAGER = True           # keep True until vllm-no-enforce-eager-pp2 validates CUDA graphs

# IMPORTANT Blackwell / GB10 workarounds:
# - VLLM_USE_RAY_COMPILED_DAG=0   — compiled DAG hangs on GB10 unified memory
# - RAY_memory_monitor_refresh_ms=0  — disable Ray OOM killer
# - NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_P2P_DISABLE=1  — 10GbE inter-node
# - --disable-custom-all-reduce    — REQUIRED for TP>1 on Blackwell (vLLM IPC fast-path fails)
# These are baked into the K8s manifests in token-labs/deploy/qwen3-coder-next/

# Not used directly by run_experiment.py (K8s manifests drive deployment),
# but used as documentation and by serve_config-aware tooling.
DISABLE_CUSTOM_ALL_REDUCE = False  # set True only when TP_SIZE > 1

ENV_VARS = {
    "VLLM_USE_RAY_COMPILED_DAG": "0",
    "RAY_memory_monitor_refresh_ms": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_P2P_DISABLE": "1",
    "HF_HUB_ETAG_TIMEOUT": "60",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
}
