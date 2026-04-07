# Inference Autoresearch Program

## Objective

Maximize throughput (tokens/second) on the DGX Spark cluster while maintaining acceptable latency.

**Primary metric**: throughput (tok/s) at concurrency=32
**Secondary metrics**: TTFT p50/p99 (ms), ITL p50/p99 (ms/tok)
**Pass criteria**: ≥3% throughput improvement without TTFT or ITL regressing >20%

## Cluster

- spark-01: nvidia@192.168.1.76 — GB10 Grace Blackwell, 128GB unified memory (NVLink-C2C), leader node
- spark-02: nvidia@192.168.1.77 — GB10 Grace Blackwell, 128GB unified memory (NVLink-C2C), worker node
- K8s namespace: `token-labs`
- Model cache: Longhorn PVCs (`model-cache-{experiment}-spark-01/02`)
- Manifests: `~/src/github.com/elizabetht/token-labs/deploy/`
- Serving endpoint: `http://192.168.1.200:8000` (LoadBalancer)
- Baseline config: `autoresearch/serve_config.py` (all defaults; experiments override specific fields)

## Workflow

1. Scheduler reads `queue.yaml`, selects highest-priority `queued` experiment
2. Deploys the experiment via `run_experiment.py` (applies K8s manifests, waits for ready)
3. Runs `benchmark.py` against the serving endpoint
4. Parses results, appends to `results.tsv`, updates `LEADERBOARD.md`
5. Marks experiment `done` or `failed` in queue.yaml
6. Commits and pushes; sends Telegram notification

## Queue Rules

- **One variable at a time**: each experiment changes exactly one thing from baseline
- **No retries**: failed experiments (OOM, timeout, crash) are marked `failed`, not re-queued with same config
- **Isolation**: shut down previous serving pod before launching next
- **Timeouts**: 30min server startup (90min for cold GB10 with JIT kernel compilation), 15min per benchmark run, 90min total per experiment
- **Approval required** for experiments that require new PVC provisioning (>300GB)

## Keep/Discard Rule

- **KEPT** — throughput improved >3% AND TTFT/ITL didn't regress >20%
- **DISCARDED** — throughput regressed OR TTFT/ITL got much worse
- **CRASHED** — experiment failed (no valid metrics)
- **TIMEOUT** — experiment hit the safety-net cap; do NOT retry same config — pick a different hypothesis

## What You Can Modify

- `autoresearch/queue.yaml` — add, remove, reprioritize experiments
- `autoresearch/serve_config.py` — the baseline all experiments diverge from
- `~/src/github.com/elizabetht/token-labs/deploy/` — K8s manifests for new experiments

## What You Cannot Modify

- `autoresearch/benchmark.py` — immutable ground truth
- `autoresearch/scheduler.py` — runtime infrastructure
- `autoresearch/run_experiment.py` — deployment logic
- `autoresearch/program.md` — these are your instructions

## Optimization Targets (prioritized by expected impact)

Based on sara4dev/inference-research results (SM120/SM121 Blackwell, 2026-04):

1. **CUDA graphs** (`remove --enforce-eager`) — sara4dev: +19% on SM120 (3553→4241 tok/s). The single biggest
   easy win. May behave differently on GB10 unified memory — watch for hangs.

2. **FP8 weight quantization** — sara4dev's largest single gain (+65%, 3553→5863 tok/s). Our model is already
   FP8-quantized (Qwen/Qwen3-Coder-Next-FP8), so this is partially captured in baseline. The remaining delta
   is FP8 KV cache (--kv-cache-dtype fp8), which sara4dev tested separately (+5% on SM120).

3. **SGLang baseline** — sara4dev: SGLang beats vLLM at 3B with defaults (+21%, 4291 vs 3553). RadixAttention
   + piecewise CUDA graph scheduler. The gap may be larger on GB10 where vLLM eager has more overhead.

4. **Prefix caching / RadixAttention** — sara4dev: prefix-caching-off was throughput-neutral (4239 vs 4241)
   meaning APC overhead is negligible. For code workloads (Qwen3-Coder-Next) with system prompts,
   expect real TTFT benefit.

5. **Chunked prefill** — sara4dev research cites +50% from TNG 2025. sara4dev didn't test this directly.
   Reduces TTFT p99 at high concurrency; may regress throughput slightly.

6. **Speculative decoding (EAGLE-3)** — NeurIPS'25 confirms 2-6x speedup in latency-sensitive scenarios.
   At high concurrency, gains diminish (GPUs better utilized). Good for TPOT/ITL improvement.

7. **Dynamo disaggregated PD** — sara4dev data: DOES NOT beat single-GPU for small models at low concurrency
   (3412-3540 vs 3553 baseline, same or worse). TTFT is significantly worse (94-124ms vs 38ms). Only consider
   for large models at very high concurrency where prefill becomes the bottleneck.

## Known DGX Spark / GB10 Constraints

- `VLLM_USE_RAY_COMPILED_DAG=0` — compiled DAG hangs on GB10 unified memory
- `RAY_memory_monitor_refresh_ms=0` — disable Ray OOM killer
- `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_P2P_DISABLE=1` — 10GbE inter-node
- SM121 (compute cap 12.1) — some kernels fall back to PTX; native SM121 kernels ~10-20% faster when available
- `--enforce-eager` may be needed if CUDA graph capture fails on aarch64
- `--disable-custom-all-reduce` REQUIRED for TP>1 on Blackwell — vLLM IPC fast-path fails on first inference
  (sara4dev confirmed; applies to SM120 and SM121 alike)
- gb10 unified memory means GPU_MEMORY_UTILIZATION semantics differ — 0.75 is the safe default for PP=2

## Strategy Tips (from sara4dev learnings)

- **CUDA graphs first** — it's the biggest easy win (+19% on Blackwell) and unblocks accurate measurement of other optimizations (eager mode masks other effects)
- **SGLang early** — the framework itself is a 20%+ win on Blackwell; comparing vLLM experiments to vLLM baseline and SGLang experiments to SGLang baseline
- **FP8 KV cache separate from FP8 weights** — sara4dev shows these have independent effects
- **TP=2 disappoints on small models** — sara4dev saw 3B model LOWER throughput with TP=2 (3233 vs 3553) due to all-reduce overhead dominating. Only expect TP=2 gains when the model barely fits in single-GPU memory
- **Dynamo disagg: don't bother for low-concurrency** — sara4dev data shows disagg adds TTFT latency without throughput gain for small models. Worth revisiting if we move to larger models or very high concurrency (>128)
- **Don't retry timeouts/crashes** — the hardware is telling you no. Change the hypothesis instead
- **Combine winners last** — run all individual experiments first; then combine the confirmed winners in one `combined-best` entry
