---
title: Cold Start & TTFT Reduction — vLLM, SGLang, TRT-LLM on DGX Spark
date: 2026-04-15
tags: [inference, vllm, sglang, trtllm, ttft, cold-start, dgx-spark]
status: research
---

# Cold Start & TTFT Reduction: Production LLM Inference on DGX Spark

## Executive Summary

Two distinct latency clocks: **pod startup latency** (seconds to minutes, dominated by weight loading and CUDA graph capture) and **per-request TTFT** (milliseconds, dominated by prefill compute, scheduler queuing, KV cache state). They require entirely different interventions.

---

## Part 1: Pod Startup / Model Load Latency

### 1.1 fastsafetensors Weight Loading
- **Flag:** `--load-format fastsafetensors`
- **Impact:** 4.8–7.5x faster weight loading. Llama 3.1 8B: 33s → ~7s
- **How:** DMA's weight blobs directly into GPU memory in parallel, bypasses CPU deserialization
- **Availability:** vLLM 0.9.0 ✅ | SGLang: No | TRT-LLM: N/A

### 1.2 CUDA Graph Capture Reduction
- **Flag:** `--cuda-graph-sizes 1,2,4,8,16,24,32,64`
- **Impact:** 54s → 7s capture time (87% reduction). Overall startup: 294s → 82s
- **How:** Skips rarely-used batch sizes
- **Availability:** vLLM 0.9.0 ✅

Also relevant:
- `torch_compile_cache` at `~/.cache/vllm/torch_compile_cache` — reused on warm pod restarts (mount PVC)
- `-O0` flag skips compilation entirely (fast start, ~30% slower inference)

### 1.3 TRT-LLM Engine File Caching
- **How:** Pre-build `.engine` file once, load in seconds on subsequent starts
- **Impact:** 10–45min build → ~10–15s load
- **Key:** Engines are deterministic per GPU SKU + driver version (DGX Spark GB10 specific)
- **Pattern:** Store on PVC or bake into container layer

### 1.4 SGLang DeepGEMM Pre-compilation
- **Command:** `python3 -m sglang.compile_deep_gemm --model meta-llama/Llama-3.1-8B --tp 1`
- **Impact:** Eliminates 10–20min JIT on first start
- **Pattern:** Run as K8s init container

### 1.5 K8s Readiness / Startup Probe Tuning
- **Problem:** vLLM `/health` returns 200 before model is loaded. K8s routes traffic immediately.
- **Fix:** Use `startupProbe` with 10min window, check `/v1/models` (only ready after model loaded)

```yaml
startupProbe:
  httpGet:
    path: /v1/models
    port: 8000
  failureThreshold: 60
  periodSeconds: 10
  initialDelaySeconds: 30
```

---

## Part 2: Per-Request TTFT

### 2.1 Chunked Prefill
- **Flags:** `--enable-chunked-prefill`, `--max-num-batched-tokens 2048`
- **Impact:** Reduces median/P99 TTFT under load by preventing long-prefill HOL blocking
- **Tradeoff:** Slightly increases TTFT at low QPS; inverts at high QPS
- **vLLM V1:** Enabled by default. SGLang: `--chunked-prefill-size 2048`

### 2.2 Prefix Caching / RadixAttention
- **Impact:** Up to 87% TTFT reduction at 87% cache hit rate
- **vLLM:** `--enable-prefix-caching` (default in V1)
- **SGLang:** Enabled by default (RadixAttention)
- **Critical unlock:** Prefix-aware routing. Round-robin LB kills cache hit rates to ~0%. Options:
  - vLLM Router (Rust, Dec 2025): consistent hashing by session ID, 25% higher throughput than llm-d
  - llm-d: scores pods by KV cache affinity
  - Simple: hash system prompt prefix → sticky session → Envoy consistent_hashing

### 2.3 Speculative Decoding (EAGLE3)
- **TTFT impact: slightly negative** (adds draft model overhead before first token)
- **Benefit:** 1.5–2.8x TPOT (decode throughput), not TTFT
- **Exception:** SpecPrefill (ICML 2025) — up to 7.66x TTFT improvement on 405B models; not yet merged into vLLM

### 2.4 CUDA Graphs for Per-Request Overhead
- **Impact:** Eliminates ~1–5ms kernel launch overhead per forward pass
- **Key:** Ensure batch size 1 is always captured (`--cuda-graph-sizes 1,...`)
- **Missing batch size:** Triggers JIT capture, adds 100–500ms one-time penalty

### 2.5 KV Cache FP8 Quantization
- **TTFT impact:** Neutral (FlashAttention-2 doesn't accelerate FP8 KV)
- **Benefit:** Memory capacity → more concurrent requests without eviction
- **Use for:** Nemotron-120B to fit more concurrent sessions
- **Flags:** `--kv-cache-dtype fp8` (vLLM/SGLang)

### 2.6 Disaggregated Prefill
- **Impact:** Decouples TTFT from TPOT optimization. Prefill pods process at full throughput, unblocked by decode queue
- **DGX Spark topology:** Prefill node-1, decode node-2, KV transfer via NVLink at 900 GB/s (~1–3ms latency)
- **vLLM flags (experimental):** `--task prefill` / `--task decode`
- **SGLang flags:** `--disaggregation-mode prefill/decode`, `--disaggregation-transfer-backend nccl|rdma`

### 2.7 Request Scheduling
- **Problem:** FCFS causes HOL blocking — one 32k-token prefill blocks 30 short requests
- **Fix:** `--scheduling-policy priority` + tag interactive requests
- **SGLang:** `--max-running-requests 200` (shed load vs. accumulate latency)

---

## Part 3: DGX Spark Specific

### NVLink Weight Loading
- DGX Spark GB10: 128 GB unified CPU+GPU coherent memory at 900 GB/s vs. 64 GB/s PCIe
- Weight loading from PVC on host: likely <10s for 7B models (vs. 33s baseline on A100)
- No traditional H2D copy — unified memory

### TP Effect on Prefill
- TP=2 across two nodes: prefill compute halved, all-reduce overhead ~2–5ms per layer
- DeepSeek-R1-7B: TP=1 single node likely faster TTFT than TP=2 (NVLink latency > compute savings at 7B)
- Nemotron-120B: TP=2 required; prefill is compute-bound so communication overhead is minor

### NVFP4 on Blackwell
- B200 FP4 tensor cores: 2x compute vs. FP8 on same silicon
- Nemotron-120B-NVFP4 fits in 128 GB single DGX Spark unified memory
- SGLang nightly has better MXFP4 kernel paths than vLLM 0.9.0 as of Q1 2026

---

## Experiment Queue (Priority Order)

| # | Experiment | Impact | Complexity | Stack |
|---|---|---|---|---|
| 1 | `--cuda-graph-sizes 1,2,4,8,16,32,64` | 54s→7s capture | Low | vLLM |
| 2 | Persist torch_compile_cache on PVC | Eliminates recompile | Low | vLLM |
| 3 | `--load-format fastsafetensors` | 4–7x weight load | Low | vLLM |
| 4 | startupProbe `/v1/models` + 10min window | No premature routing | Low | All |
| 5 | SGLang DeepGEMM pre-compile init container | Eliminates 10–20min JIT | Medium | SGLang |
| 6 | TRT-LLM engine PVC cache | Eliminates build | Medium | TRT-LLM |
| 7 | Prefix caching + prefix-aware routing | 88% TTFT reduction | Medium | vLLM |
| 8 | `--max-num-batched-tokens 2048` tuning | Reduce HOL blocking | Low | vLLM, SGLang |
| 9 | Priority scheduling for interactive vs. batch | Protect TTFT under load | Medium | vLLM, SGLang |
| 10 | KV cache FP8 + FlashInfer backend | More KV headroom | Low | vLLM |
| 11 | Disaggregated prefill on dual-node | Decouple TTFT/TPOT | High | vLLM, SGLang |
| 12 | EAGLE3 speculative decoding | 1.5–2.8x TPOT | Medium | vLLM, SGLang |

---

## Availability Matrix

| Technique | vLLM 0.9.0 | SGLang nightly | TRT-LLM latest |
|---|---|---|---|
| Chunked prefill | Yes (default V1) | Yes | Yes |
| Prefix caching | Yes (default V1) | Yes | Yes |
| KV cache FP8 | Yes | Yes | Yes (INT8/FP8) |
| CUDA graph control | Yes | Yes | Managed internally |
| fastsafetensors | Yes | No | N/A |
| torch.compile cache | Yes (auto) | N/A | N/A |
| Disaggregated prefill | Yes (experimental) | Yes (production) | Partial |
| Speculative decoding | Yes (EAGLE3) | Yes | Yes (Medusa) |
| Priority scheduling | Yes | Yes | No |
| DeepGEMM pre-compile | N/A | Yes | N/A |
