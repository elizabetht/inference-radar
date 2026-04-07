# LLM Inference Optimization: Deep Dive Research Report

> Date: 2026-03-28
> Methodology: Systematic survey of 2024-2026 papers, framework codebases, production benchmarks, and hardware roadmaps. Modeled after Karpathy's autoresearch approach.
>
> **DGX Spark adaptation note**: Adapted from sara4dev/inference-research. This lab runs 2x GB10 Grace
> Blackwell (SM121, arm64, 128GB unified memory), PP=2 across spark-01/spark-02, model: Qwen/Qwen3-Coder-Next-FP8.
> sara4dev's hardware was RTX PRO 6000 Blackwell (SM120, x86_64). Key differences for GB10: unified memory
> means no separate VRAM limit; SM121 vs SM120 means some kernel paths differ; ARM64 means TRITON_PTXAS_PATH
> workarounds may be needed for SGLang.

---

## 1. Executive Summary

LLM inference optimization has undergone a phase transition since 2024. The field has moved from single-technique improvements to **full-stack, system-level co-design** spanning hardware, kernels, scheduling, and orchestration. Here are the top findings:

### Key Takeaways

1. **Disaggregation is now table stakes.** Every production framework (vLLM, SGLang, TensorRT-LLM, Dynamo, llm-d) implements prefill-decode disaggregation. The frontier has moved to **attention-FFN disaggregation** and **decode-specialized ASICs**.

2. **The memory wall is the #1 bottleneck.** H100 can theoretically do 62,000 tok/s for an 8B model but practical systems barely hit 200 tok/s. The gap is entirely memory bandwidth. Solutions: speculative decoding (shift memory-bound to compute-bound), KV cache compression, CXL memory pooling, and hardware innovations like High Bandwidth Flash (HBF).

3. **Speculative decoding has matured.** EAGLE-3 (NeurIPS'25) achieves 2-6x speedups. SuffixDecoding (NeurIPS'25 Spotlight) achieves 5.3x on agentic workloads with zero GPU overhead. Speculative Speculative Decoding (ICLR'26) parallelizes speculation and verification for 2-5x over standard speculative decoding.

4. **SGLang is faster than vLLM.** On H100, SGLang delivers ~16,200 tok/s vs vLLM's ~12,500 tok/s (29% gap). This holds even when vLLM uses identical FlashInfer kernels. The difference is scheduling overhead, RadixAttention, and zero-overhead batch scheduling.

5. **MoE inference is the defining challenge of 2026.** DeepSeek-R1 (671B, 256 experts) requires 13,719 GB/s memory bandwidth at full activation. Expert parallelism with DeepEP, wide EP on NVL72, and expert caching/offloading are all critical. This is where the most impactful optimization work will happen.

6. **Agentic workloads are structurally different.** Multi-turn agents accumulate 5-10x more input tokens by turn 30. Cross-call KV cache reuse, workflow-aware scheduling (Helium), and speculative tool execution are nascent areas with huge potential.

### Top 5 Opportunities (Impact x Feasibility)

| Rank | Opportunity | Expected Impact | Difficulty |
|------|------------|-----------------|------------|
| 1 | SuffixDecoding for agentic workloads in vLLM/SGLang | 2-5x latency reduction | Medium |
| 2 | Attention-FFN disaggregation for MoE models | 30-50% throughput gain | High |
| 3 | Hierarchical KV cache with CXL memory pooling | 3-6x effective cache size | High |
| 4 | Workflow-aware scheduling for agentic inference | 1.5-4x speedup on multi-call workflows | Medium |
| 5 | Inter-node all-reduce optimization (NVRAR-style) for decode | 10-30% decode latency reduction | Medium |

---

## 2. Research Paper Survey

### 2.1 Speculative Decoding

Speculative decoding is the single most effective technique for reducing per-request latency. It shifts decode from memory-bound to compute-bound by drafting multiple tokens with a lightweight model and verifying them in a single forward pass.

#### EAGLE Family (ICML'24 / EMNLP'24 / NeurIPS'25)

| Version | Venue | Core Idea | Speedup |
|---------|-------|-----------|---------|
| EAGLE-1 | ICML'24 | Lightweight draft head reusing target model features | 2-3x |
| EAGLE-2 | EMNLP'24 | Dynamic tree structure via confidence-based acceptance approximation | 2.5-4x |
| EAGLE-3 | NeurIPS'25 | Training-time testing + multi-layer feature fusion (low/mid/high) | 2-6.5x (1.4x over EAGLE-2) |

**Key insight from EAGLE-3**: Instead of predicting features of the next token (which creates a distribution mismatch between training and inference), EAGLE-3 uses "training-time testing" to simulate actual inference conditions during training. It also fuses features from multiple layers rather than just the top layer, capturing richer semantic information.

**Adoption**: Integrated into vLLM and SGLang. vLLM Q1 2026 roadmap includes stable MTP support for Qwen3-Next and DeepSeek, EAGLE-3 testing, and frontier model speculator releases on HuggingFace.

**Throughput at scale**: At batch size 64 in SGLang, EAGLE-3 achieves 1.38x throughput improvement (spec decoding benefits diminish at high batch sizes because the GPU is already better utilized).

#### Sequoia (NeurIPS'24)

**Core technique**: Dynamic programming algorithm to find optimal tree structure for speculated tokens, plus hardware-aware tree optimizer that automatically selects tree size and depth for target hardware.

**Results**: Up to 4.04x speedup on Llama2-7B (A100). Key contribution is making speculative decoding **scalable** to larger speculation budgets by optimizing the tree topology.

#### SuffixDecoding (NeurIPS'25 Spotlight)

**Core technique**: Builds suffix trees over previously generated token sequences. At each step, matches recent output against the suffix tree to find likely continuations, verified in a single forward pass. Maintains two trees: global (historical outputs) and per-request (current context).

**Results**: Up to 5.3x speedup, outperforming EAGLE-2/3 by 2.8x and Token Recycling by 1.9x.

**Critical advantage**: Runs entirely on CPU memory, zero GPU overhead. Particularly effective for agentic workloads with repetitive patterns (code generation, structured outputs, tool-calling). This is a **massively underexploited technique**.

#### Speculative Speculative Decoding (ICLR'26)

**Core technique**: Parallelizes speculation and verification. While the verifier checks round T tokens, the speculator predicts likely verification outcomes and pre-prepares speculations for round T+1, stored in a "speculation cache". If the actual outcome matches a predicted one, the next speculation is returned immediately.

**Results**: Up to 2x faster than optimized speculative decoding, 5x faster than autoregressive. Authors: Tanishq Kumar (Stanford), Tri Dao (Princeton), Avner May (Together AI).

**Significance**: This is the first paper to address the sequential bottleneck *within* speculative decoding itself. Requires separate hardware for speculator and verifier, fitting naturally into disaggregated architectures.

#### P-EAGLE (AWS, 2025)

Parallel speculative decoding variant integrated into vLLM. Enables parallel verification of draft tokens, reducing the sequential dependency in the verification step.

### 2.2 KV Cache Optimization

The KV cache is the largest memory consumer during inference, taking up to 70% of GPU memory for long-context workloads.

#### Quantization Approaches

| Method | Bits | Key Innovation | Result |
|--------|------|----------------|--------|
| KIVI (ICML'24) | 2-bit | Tuning-free asymmetric quantization | Near-lossless at 2-bit |
| KVQuant (NeurIPS'24) | 2-bit | Per-channel key quantization + pre-RoPE quantization + non-uniform quantization | 8x compression, 1M context on single A100 |
| Coupled Quantization | 1-bit | Joint quantization of K and V | Near-lossless at 1 bit per channel |
| OTT (ACL'25) | 2-bit | Outlier token tracing | Outperforms KIVI on GSM8K, BBH, HumanEval |

**Production status**: vLLM supports FP8 KV cache quantization. Deeper quantization (2-bit, 1-bit) remains research-stage but shows enormous promise for long-context serving.

#### Compression and Eviction

| Method | Technique | Result |
|--------|-----------|--------|
| ChunkKV | Semantic chunk-level compression (preserves linguistic structures) | Maintains quality with significant compression |
| KVTC (KV Transform Coding) | Transform coding for KV cache | 20-40x compression maintaining reasoning accuracy |
| DynamicKV | Per-layer token budgets based on attention patterns | 85% of full-KV performance at 1.7% memory |
| TokenSelect (EMNLP'25) | Interleaved token selection for sparse attention | Efficient 100K+ context |

**The frontier**: KVTC's 20-40x compression is remarkable. If this holds in production, it would fundamentally change the economics of long-context serving.

#### Distributed KV Cache Systems

| System | Key Feature | Adoption |
|--------|------------|----------|
| LMCache | 8 storage backends, 4 processor types, cross-engine sharing | vLLM, SGLang, llm-d, Dynamo, KServe |
| Mooncake | KVCache-centric disaggregated architecture, GPUDirect RDMA | Production at Moonshot AI (100B+ tokens/day) |
| NIXL | Vendor-agnostic transfer library (NVLink/RDMA/GDS) | Dynamo, TensorRT-LLM, vLLM |

**Key insight**: LMCache achieves up to 15x throughput improvement by reusing KV caches across engines and queries, not just within a single prefix tree. Combined with Redis for distributed storage, it enables cluster-wide cache sharing.

### 2.3 Attention Optimization

#### FlashAttention-3 (2024-2025)

- **Performance**: 1.5-2x faster than FA-2 on H100, up to 740 TFLOPS (75% H100 utilization)
- **FP8 support**: Reaches ~1.2 PFLOPS with 2.6x smaller error than baseline FP8 attention
- **Key techniques**: Warp specialization, TMA (Tensor Memory Accelerator), interleaved matmul/softmax pipeline
- **Adoption**: Universal — every serving framework uses FA-3 on Hopper/Blackwell

#### Context Parallelism for Long Context

- **Ring Attention variants**: Pass-KV and Pass-Q cover prefill and decode use cases
- **Performance**: Near-linear scaling for prefill with up to 128 H100s (16 nodes). 1M context prefill in 77s, 128K in 3.8s with Llama3 405B
- **Challenge**: Communication-to-compute ratio worsens at scale — computation decreases quadratically per step while communication only decreases linearly
- **Meta's approach**: N-D parallelism combining CP, PP, EP, and TP across nodes

#### Token Sparse Attention (2026)

Interleaved token selection reduces attention computation for long contexts while maintaining quality. Key for making 100K+ token inference economically viable.

### 2.4 Prefill-Decode Disaggregation

#### DistServe (OSDI'24) — The Paper That Started It All

- 7.4x more requests served or 12.6x tighter SLO compliance vs prior systems
- Core insight: prefill is compute-bound, decode is memory-bound — they need different GPU configurations

#### Current State (18 Months Later)

From the DistServe team's retrospective (late 2025):
- **Universally adopted**: Every production framework implements it
- **The frontier**: Attention-FFN Disaggregation (AFD) — separating attention operations (memory-bound) from FFN/MoE computations (compute-bound) within the decode phase itself
- **Hardware implications**: Companies (Huawei, Enflame, Biren) are building decode-specialized ASICs, showing disaggregation now drives hardware design
- **Splitwise**: 1.4x throughput at 20% lower cost, or 2.35x throughput at same cost using heterogeneous hardware (H100 prefill + A100 decode)

#### DuetServe (2025)

Fine-grained SM-level partitioning within a single GPU, enabling prefill and decode to share a GPU without head-of-line blocking. Useful for smaller deployments where full disaggregation isn't cost-effective.

### 2.5 Quantization for Inference

#### Weight Quantization

| Method | Format | Key Innovation | Throughput Impact |
|--------|--------|----------------|-------------------|
| AWQ | INT4 | Activation-aware weight quantization (protects salient channels) | ~3x over BF16 |
| GPTQ | INT4 | Layer-wise optimal quantization | ~3x over BF16 |
| Marlin kernels | INT4 (AWQ/GPTQ) | Optimized GPU kernels for quantized inference | 10.9x speedup for AWQ, 2.6x for GPTQ vs naive |
| NVFP4 | FP4 | NVIDIA's 4-bit float for Blackwell (1 sign, 2 exp, 1 mantissa) | Hardware-accelerated on Blackwell |
| MXFP4 | FP4 | Microscaling FP4 | Supported in TensorRT-LLM and ROCm 7.0 |

**Production reality**: FP8 is the current production standard on Hopper. FP4 is Blackwell-native. AWQ and GPTQ with Marlin kernels dominate for cost-sensitive INT4 serving.

**GB300 NVL72 headline**: 50x throughput per megawatt and 35x lower cost per token vs Hopper, largely due to NVFP4 hardware acceleration.

### 2.6 Batching and Scheduling

#### Continuous Batching + Chunked Prefill

Now universal. vLLM's continuous batching achieves 60+ tok/s per user on A100 for long context vs 15-20 tok/s with HuggingFace Transformers (3-4x improvement).

#### FlowPrefill (2026)

Introduces **operator-level preemption** — fine-grained interruption at operator boundaries — and **event-driven scheduling** that triggers only on request arrival/completion. Eliminates head-of-line blocking from large prefills without the overhead of fixed-size chunking.

#### SARATHI

Chunked prefills + decode-maximal batching: fills GPU bubbles during decode with chunks of prefill work, raising utilization and eliminating pipeline stalls.

### 2.7 MoE Inference Optimization

MoE is the defining architecture of 2026. DeepSeek-R1 (671B params, 37B active, 256 experts), Llama 4, Mistral Large 3, and Gemini all use MoE.

#### Expert Parallelism

| System | Innovation | Result |
|--------|-----------|--------|
| DeepEP | Dual-mode dispatch (Normal for prefill, Low-Latency for decode), pure RDMA, hook-based overlapping | Industry-standard EP library |
| Wide EP on vLLM + llm-d | Expert parallelism across many GPUs | 2.2k tok/s/H200 for DeepSeek |
| LMSYS 96-GPU deployment | PD disaggregation + large-scale EP | 52.3k input tok/s, 22.3k output tok/s per node |

#### The MoE Paradox

MoE models use fewer FLOPs per token than dense equivalents, but require **far more memory** because every expert must be accessible for dynamic routing, even though most stay idle. DeepSeek-R1 at full activation needs 13,719 GB/s bandwidth — exceeding even GB200 NVL72's capabilities.

#### Expert Caching and Offloading

On memory-constrained systems, expert offloading to CPU/SSD is essential. The sparse activation pattern (only ~8/256 experts active per token in DeepSeek) enables predictive expert loading. This is an active area — better expert prediction and prefetching could significantly reduce latency.

### 2.8 Structured Generation

#### XGrammar (MLSys'25)

- Splits vocabulary into context-independent (precomputed) and context-dependent (runtime) tokens
- **10x faster** than prior constrained decoding engines
- **<40 microseconds** per token for mask generation
- Near-zero overhead when integrated with serving frameworks on H100
- **Adopted by**: vLLM (default), SGLang

#### XGrammar-2 (2026)

Extension for agentic LLMs with dynamic grammar switching during generation (e.g., switching between JSON, code, and natural language within a single response).

#### llguidance (Microsoft)

Rust-based engine: ~50 microseconds CPU time per token for 128K vocabulary. Negligible startup cost.

**Bottom line**: Structured generation overhead is essentially solved. XGrammar and llguidance make constrained decoding free.

### 2.9 Diffusion LLMs

A paradigm shift from autoregressive to parallel token generation.

| System | Technique | Result |
|--------|-----------|--------|
| LLaDA / Dream / Gemini Diffusion | Masked diffusion with bidirectional attention | Approaching autoregressive quality |
| Fast-dLLM | Block-wise KV cache for diffusion models | Enables cache reuse in bidirectional models |
| Learn2PD | Learnable parallel decoding filter | 31.23 TPS, 57.51x speedup with Dual Cache |

**Current status**: Diffusion LLMs are approaching autoregressive quality but practical inference speed still lags due to lack of KV cache and quality degradation with aggressive parallel decoding. Watch this space — if quality parity is achieved, diffusion LLMs break the memory wall entirely by making decode compute-bound.

### 2.10 Network Optimization

#### NCCL 2.27

- Symmetric memory support: up to 7.6x latency reduction for small messages
- Direct NIC support: full network bandwidth by bypassing CPU

#### NVRAR (Beyond NCCL)

GPU-initiated all-reduce for decode's small message sizes in multi-node settings. Hierarchical approach: NCCL intra-node, custom recursive all-reduce inter-node. Critical for multi-node decode which NCCL handles poorly.

#### MSCCL++

Up to 3.5x faster than NCCL for small messages, 1.6x for large messages. Programmable communication schedules.

#### DDA on AMD MI300X

Achieved performance parity with H100 NCCL. 10-50% decode speedup over RCCL baseline.

---

## 3. Framework Feature Matrix

| Feature | vLLM | SGLang | TensorRT-LLM | Dynamo | llm-d |
|---------|------|--------|---------------|--------|-------|
| **Continuous batching** | Yes | Yes | Yes (in-flight) | Yes (via engines) | Yes (via vLLM) |
| **PagedAttention** | Yes (v1) | Yes | Variant | Via engines | Via vLLM |
| **FlashAttention-3** | Yes | Yes | Yes | Via engines | Via vLLM |
| **P/D disaggregation** | Yes | Yes | Yes | Yes (automated) | Yes |
| **Speculative decoding** | EAGLE-1/2/3, Medusa, draft model | EAGLE-3, draft model | EAGLE, Medusa, draft model, MTP | Via engines | Via vLLM |
| **FP8 inference** | Yes | Yes | Yes (best) | Via engines | Via vLLM |
| **FP4/NVFP4** | In progress | Planned | Yes | Via TRT-LLM | Via vLLM |
| **KV cache quantization** | FP8 | FP8 | FP8 | Via engines | Via vLLM |
| **Structured output** | XGrammar | XGrammar + compressed FSM | Limited | Via engines | Via vLLM |
| **Prefix caching** | Basic radix | RadixAttention (best) | Basic | KV-aware router | Prefix-cache aware routing |
| **KV cache offloading** | CPU | CPU | CPU | GPU/CPU/SSD/S3 (KVBM) | GPU/CPU/SSD/remote |
| **Multi-node** | Yes | Yes | Yes | Yes (best orchestration) | Yes (K8s-native) |
| **Expert parallelism** | Wide EP | EP | EP | EP + orchestration | Wide EP |
| **Context parallelism** | Basic | Basic | Full | Via engines | Via vLLM |
| **Hardware support** | NVIDIA, AMD, TPU, CPU | NVIDIA, AMD | NVIDIA only | NVIDIA (primary) | NVIDIA, AMD, Intel, TPU |
| **KV-aware routing** | No (via vllm-router) | RadixAttention | No | Smart Router (Radix Tree) | Prefix-cache aware scheduler |
| **Auto-configuration** | Manual | Manual | Manual | AIConfigurator (10K configs in 30s) | Manual |
| **SLO-driven autoscaling** | No | No | No | Planner (80% fewer SLA breaches) | Variant Autoscaler |
| **K8s native** | No | No | No | Grove operator | Yes (core design) |
| **Weight streaming** | No | No | No | ModelExpress (7x faster cold start) | No |

### Framework Positioning

- **vLLM**: Largest community, most model support, broadest hardware. Trailing SGLang on raw performance but closing the gap. Best for multi-hardware deployments and RL integration.
- **SGLang**: Fastest single-engine performance (29% over vLLM). Best for structured generation, prefix-heavy workloads (RAG, multi-turn), and latency-sensitive applications.
- **TensorRT-LLM**: Best single-GPU performance on NVIDIA hardware. FP8/FP4 are first-class. Complex to deploy and update.
- **NVIDIA Dynamo**: Best orchestration layer. Adds KV-aware routing, SLO-driven autoscaling, and automated configuration on top of any engine. Vendor lock-in concern.
- **llm-d**: Best Kubernetes integration. Vendor-neutral, CNCF Sandbox. Less mature than Dynamo but growing fast with Red Hat, Google, IBM, and NVIDIA backing.

### What Each Framework Is Missing

**vLLM lacks**:
- Efficient KV-aware routing (relies on external vllm-router)
- SLO-driven autoscaling
- Automated configuration tuning
- Native SuffixDecoding integration
- Performance parity with SGLang (29% gap)

**SGLang lacks**:
- Broad hardware support beyond NVIDIA/AMD
- Enterprise-grade orchestration (no K8s operator)
- Disaggregated serving maturity (vLLM is ahead here)
- RL training integration

**TensorRT-LLM lacks**:
- Non-NVIDIA hardware support
- Flexible structured generation
- Easy deployment (compilation step is painful)
- Community velocity

**Dynamo lacks**:
- Broad hardware support (NVIDIA-centric)
- CNCF/open governance
- Community transparency (NVIDIA-controlled roadmap)

**llm-d lacks**:
- Engine diversity (vLLM only, no SGLang/TRT-LLM)
- Automated configuration (no AIConfigurator equivalent)
- Weight streaming for fast model swapping
- Maturity on Blackwell

---

## 4. Gap Analysis: Unsolved Problems Ranked by Impact

### 4.1 [CRITICAL] Agentic Workflow Optimization

**The problem**: Current serving frameworks optimize individual LLM calls. Agentic workflows make 50-200 calls per task with accumulated context (5-10x more input tokens by turn 30). Cross-call optimization is almost entirely missing.

**What exists**: Helium (research, 2026) achieves 1.34-39.5x speedups via workflow-aware scheduling, proactive KV caching, and prompt cache substitution. KVFlow abstracts workflows as Agent Step Graphs to prevent premature cache eviction. SuffixDecoding exploits repetitive patterns in agentic output.

**What's missing in production**:
- No framework has workflow-level scheduling
- No framework does speculative tool execution (overlap LLM inference with tool calls)
- KV cache eviction policies are unaware of multi-turn session structure
- No cross-call prompt deduplication

**Impact**: Agentic workloads are the fastest-growing inference use case. 2-10x cost reduction is achievable.

### 4.2 [CRITICAL] MoE Expert Management at Scale

**The problem**: MoE models dominate frontier capabilities but their memory requirements are extreme. DeepSeek-R1 needs all 256 experts accessible despite only ~8 being active per token. Expert parallelism communication is the bottleneck.

**What exists**: DeepEP for all-to-all communication, wide EP in vLLM/llm-d, expert offloading for edge.

**What's missing**:
- Predictive expert prefetching (use routing predictions to pre-load experts)
- Expert popularity-aware placement (hot experts on fast memory, cold on CPU/SSD)
- Cross-request expert batching (group requests using same experts)
- Adaptive expert parallelism (change EP degree based on workload)
- Expert pruning during inference (skip consistently unused experts)

**Impact**: 20-50% throughput improvement for MoE models. Direct cost savings on the most expensive models.

### 4.3 [HIGH] Deep KV Cache Quantization in Production

**The problem**: Research shows 2-bit and even 1-bit KV cache quantization with near-lossless quality (KIVI, KVQuant, Coupled Quantization). Production frameworks only support FP8 KV cache.

**What's missing**: Production-quality 2-bit KV cache in vLLM/SGLang. This would 4x the effective KV cache size, enabling 4x longer contexts or 4x more concurrent requests.

**Impact**: 4x memory reduction for KV cache. Transforms the economics of long-context serving.

### 4.4 [HIGH] Attention-FFN Disaggregation for MoE

**The problem**: Within the decode phase, attention is memory-bound while FFN/MoE is compute-bound. Current P/D disaggregation only splits prefill from decode. AFD splits *within* decode.

**Who's working on it**: DistServe team mentioned it as the next frontier. No production implementation exists.

**Impact**: 30-50% throughput improvement for large MoE models by better hardware utilization of both phases.

### 4.5 [HIGH] SuffixDecoding for Production Agentic Workloads

**The problem**: SuffixDecoding achieves 5.3x speedup with zero GPU overhead, outperforming EAGLE-2/3 by 2.8x. It runs on CPU. Yet no production framework integrates it.

**Why it matters**: Agentic code generation, structured output, and tool-calling workloads have highly predictable token sequences. SuffixDecoding is purpose-built for this.

**Implementation path**: Build a CPU-side suffix tree that indexes previous outputs. At each decode step, match recent tokens, propose continuations, verify in batch. Integrate with vLLM/SGLang's speculative decoding infrastructure.

### 4.6 [HIGH] CXL Memory Pooling for KV Cache

**The problem**: GPU HBM is expensive and limited. CPU DRAM is cheap but slow over PCIe. CXL provides DRAM-like latency at higher capacity.

**Results**: CXL achieves 3.8x speedup over 200G RDMA and 6.5x over 100G RDMA for KV cache access. Dramatic TTFT reduction.

**What's missing**: No inference framework has CXL-aware memory management. No tiered caching policy that intelligently places KV blocks across GPU HBM, CXL-attached DRAM, and host DRAM based on access patterns.

### 4.7 [MEDIUM] Heterogeneous GPU Cluster Scheduling

**The problem**: Real-world clusters have mixed GPU generations (A100, H100, H200, B200). Current frameworks assume homogeneous hardware.

**What exists**: Helix (ASPLOS'25) formulates LLM inference over heterogeneous GPUs as a max-flow problem, achieving 3.3x throughput improvement. Splitwise uses H100 for prefill and A100 for decode.

**What's missing**: No production framework has heterogeneous-aware scheduling. Dynamo's Planner is closest but still primarily targets homogeneous pools.

### 4.8 [MEDIUM] Inter-Node Decode Latency

**The problem**: NCCL's all-reduce performance degrades significantly for small messages across nodes — exactly the pattern in distributed decode. This is a ~10-30% latency penalty for multi-node tensor-parallel decode.

**What exists**: NVRAR achieves significant improvements with GPU-initiated, hierarchical all-reduce. MSCCL++ is 3.5x faster than NCCL for small messages.

**What's missing**: These optimized collective implementations aren't integrated into vLLM/SGLang's communication backends. They rely on stock NCCL.

### 4.9 [MEDIUM] Long-Context Inference Efficiency

**The problem**: 100K+ token contexts are common for document analysis and code understanding. Quadratic attention cost makes prefill expensive, and large KV caches consume memory during decode.

**What exists**: Context parallelism (near-linear scaling to 128 GPUs), token sparse attention, ring attention variants.

**What's missing**: Production-quality context parallelism in vLLM/SGLang (both have "basic" support). Adaptive sparse attention that adjusts token selection based on query complexity. Integration of KV cache compression (KVTC, DynamicKV) with context parallelism.

### 4.10 [MEDIUM] Speculative Decoding at High Batch Sizes

**The problem**: Speculative decoding benefits diminish at high batch sizes because the GPU is already compute-saturated. EAGLE-3 achieves only 1.38x at batch size 64 vs 6.5x at batch size 1.

**What's needed**: Batch-aware speculative decoding that adjusts draft length and tree width based on current GPU utilization. At low utilization, speculate aggressively. At high utilization, reduce or disable speculation.

---

## 5. Concrete Proposals

### Proposal 1: SuffixDecoding Integration for Agentic Inference

- **What**: Integrate SuffixDecoding into vLLM and SGLang as a CPU-side speculative decoding backend, optimized for agentic and structured-output workloads
- **Why**: 5.3x speedup (NeurIPS'25 Spotlight) with zero GPU overhead. Agentic workloads have repetitive token patterns (code templates, JSON schemas, tool-call formats) that suffix trees capture perfectly
- **How much**: 2-5x latency reduction for agentic workloads, 1.5-3x for general structured output
- **Where**: vLLM (largest community), SGLang (fastest engine)
- **Difficulty**: Medium. The algorithm is well-described. Main challenges: (1) suffix tree memory management at scale, (2) integration with existing speculative decoding verification, (3) hybrid strategy combining SuffixDecoding with EAGLE for non-repetitive segments
- **Prior art**: SuffixDecoding paper (CMU), EAGLE-3 integration in vLLM/SGLang

### Proposal 2: Production 2-bit KV Cache Quantization

- **What**: Ship KIVI/KVQuant-style 2-bit KV cache quantization as a production feature in vLLM
- **Why**: 4x memory reduction vs FP8. Enables 4x longer contexts or 4x more concurrent requests on same hardware
- **How much**: KIVI shows near-lossless quality at 2-bit. KVQuant enables 1M context on single A100 (8x compression). OTT (ACL'25) improves further with outlier tracking
- **Where**: vLLM (has FP8 KV cache infrastructure to extend)
- **Difficulty**: Medium. KIVI and KVQuant are well-understood. Challenges: (1) kernel optimization for 2-bit dequantization during attention, (2) dynamic precision switching based on layer/head importance, (3) quality validation across model families
- **Prior art**: KIVI (ICML'24), KVQuant (NeurIPS'24), OTT (ACL'25), vLLM FP8 KV cache

### Proposal 3: Workflow-Aware Scheduling for Agentic Inference

- **What**: Build a scheduling layer that understands multi-step agentic workflows, enabling proactive KV cache management, cross-call deduplication, and speculative tool execution
- **Why**: Agentic workloads accumulate 5-10x context by turn 30. Current per-call optimization misses 60-80% of reuse opportunities
- **How much**: Helium shows 1.34-39.5x speedups. Even conservative cross-call caching yields 2-3x
- **Where**: Dynamo (has routing infrastructure), llm-d (has K8s-native scheduling)
- **Difficulty**: High. Requires: (1) workflow DAG specification format, (2) KV cache lifetime management across calls, (3) integration with existing prefix caching, (4) speculative execution of likely tool-call branches
- **Prior art**: Helium (2026 paper), KVFlow, Parrot, AgentScope

### Proposal 4: Predictive Expert Prefetching for MoE

- **What**: Use lightweight routing prediction to prefetch experts before they're needed, overlapping expert loading with computation
- **Why**: MoE decode latency is dominated by expert loading for non-resident experts. Predictive loading can hide this latency
- **How much**: Estimated 20-40% decode latency reduction for large MoE models with many experts (DeepSeek-R1 scale). Higher benefit when experts don't fit in GPU memory
- **Where**: vLLM (has MoE/EP infrastructure), TensorRT-LLM (best MoE kernels)
- **Difficulty**: Medium. Train lightweight predictor on routing patterns. Use predictions to issue async expert loads 1-2 tokens ahead. Key challenge: prediction accuracy and the cost of wrong predictions
- **Prior art**: Expert offloading literature, DeepEP's dual-mode dispatch

### Proposal 5: Custom Inter-Node All-Reduce for Decode

- **What**: Replace NCCL all-reduce with NVRAR-style GPU-initiated hierarchical all-reduce for decode's small message pattern
- **Why**: NCCL's inter-node performance degrades significantly for small messages (typical of decode all-reduce in tensor parallelism)
- **How much**: 10-30% decode latency reduction for multi-node TP deployments
- **Where**: vLLM, SGLang (both use NCCL by default)
- **Difficulty**: Medium. NVRAR implementation exists (UMD research). Integration requires replacing the communication backend for specific operations while keeping NCCL for others
- **Prior art**: NVRAR (UMD), MSCCL++ (Microsoft), DDA (AMD)

### Proposal 6: Attention-FFN Disaggregation for MoE Decode

- **What**: Split the decode phase further: attention ops on memory-optimized GPUs, FFN/MoE on compute-optimized GPUs
- **Why**: Within decode, attention is memory-bound (KV cache reads) while FFN/MoE is compute-bound (matrix multiplications). Running both on the same hardware means neither is optimally utilized
- **How much**: Estimated 30-50% throughput improvement for large MoE models. The DistServe team identified this as the next major frontier
- **Where**: Dynamo (has disaggregation infrastructure), vLLM (has P/D disaggregation)
- **Difficulty**: High. Requires: (1) efficient intermediate tensor transfer between attention and FFN GPUs, (2) pipeline scheduling to keep both sides busy, (3) hardware-aware placement optimizer
- **Prior art**: DistServe retrospective, SPAD (specialized P/D hardware, 2025)

### Proposal 7: Adaptive Batch-Aware Speculative Decoding

- **What**: Dynamically adjust speculative decoding aggressiveness based on current GPU utilization and batch size
- **Why**: Speculative decoding gives 6.5x at batch=1 but only 1.38x at batch=64. At high utilization, the extra compute for verification hurts throughput
- **How much**: 10-30% throughput improvement at moderate batch sizes (8-32) by finding the optimal speculation budget
- **Where**: vLLM, SGLang (both support EAGLE)
- **Difficulty**: Low-Medium. Monitor GPU utilization, adjust draft length and tree width. The Sequoia paper's hardware-aware optimizer provides a starting point
- **Prior art**: Sequoia (NeurIPS'24), EAGLE-3 (NeurIPS'25)

---

## 6. Open Questions

### 6.1 Will Diffusion LLMs Replace Autoregressive Decoding?

Masked diffusion models (LLaDA, Dream, Gemini Diffusion) generate tokens in parallel, fundamentally breaking the memory wall. Learn2PD achieves 57.5x speedup. But quality parity with autoregressive models remains unproven at scale. If diffusion LLMs achieve quality parity, most decode-phase optimizations become irrelevant. **Timeline estimate: 12-24 months to know.**

### 6.2 How Will Blackwell Ultra (GB300) Change the Optimization Landscape?

GB300 NVL72 delivers 50x throughput/watt over Hopper. At this performance level, which bottlenecks shift? Does the memory wall move from HBM bandwidth to inter-GPU communication? Does FP4 accuracy hold for all model families? Early data suggests 1.5x lower cost per token vs GB200 for long-context workloads.

### 6.3 Is the 29% SGLang-vLLM Gap Structural or Fixable?

SGLang's 29% throughput advantage persists even when vLLM uses identical kernels. This suggests the gap is in scheduling overhead and architecture, not kernels. vLLM's Q1 2026 roadmap targets data structure efficiency and process simplification. Can these close the gap without a full rewrite?

### 6.4 What's the Right Abstraction for KV Cache as a Service?

LMCache, Mooncake, NIXL, and Dynamo's KVBM all provide KV cache management but with different abstractions. Should KV cache be treated as a distributed cache (Redis-like), a filesystem (S3-like), or a memory tier (CXL-like)? The right abstraction affects every layer above it.

### 6.5 Can Expert Routing Be Predicted Accurately Enough for Prefetching?

If we can predict which experts will be activated 1-2 tokens ahead with >90% accuracy, expert prefetching becomes transformative. But routing decisions depend on the current token, creating a chicken-and-egg problem. Speculative decoding + routing prediction is a natural combination that hasn't been explored.

### 6.6 What's the Optimal Scheduling Granularity?

FlowPrefill proposes operator-level preemption. DuetServe proposes SM-level partitioning. Traditional systems schedule at request level. What's the right granularity for maximum throughput with SLO compliance? The answer likely depends on workload and hardware, suggesting adaptive granularity.

### 6.7 How Will Agentic Inference Change Framework Architecture?

Current frameworks are optimized for stateless request-response. Agentic workloads are stateful, multi-turn, with conditional branching. Do we need a fundamentally different serving architecture — more like a workflow engine than a request router?

---

## 7. References

### Speculative Decoding

- [EAGLE-1/2/3 Official Implementation](https://github.com/SafeAILab/EAGLE) — SafeAI Lab. ICML'24, EMNLP'24, NeurIPS'25.
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://openreview.net/forum?id=4exx1hUffq) — OpenReview, NeurIPS'25.
- [Sequoia: Scalable and Robust Speculative Decoding](https://arxiv.org/abs/2402.12374) — NeurIPS'24.
- [SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](https://suffix-decoding.github.io/) — NeurIPS'25 Spotlight.
- [Speculative Speculative Decoding](https://openreview.net/pdf?id=aL1Wnml9Ef) — ICLR'26. Kumar, Dao, May.
- [P-EAGLE: Parallel Speculative Decoding in vLLM](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — AWS, 2025.
- [How Speculative Decoding Boosts vLLM Performance by up to 2.8x](https://blog.vllm.ai/2024/10/17/spec-decode.html) — vLLM Blog.
- [Speculative Decoding Overview — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)
- [NVIDIA Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

### KV Cache Optimization

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — ICML'24.
- [KVQuant: Towards 10 Million Context Length LLM Inference](https://arxiv.org/abs/2401.18079) — NeurIPS'24.
- [ChunkKV: Semantic-Preserving KV Cache Compression](https://openreview.net/forum?id=20JDhbJqn3) — OpenReview, 2025.
- [KV Cache Transform Coding (KVTC)](https://openreview.net/forum?id=aNVKROYpLB) — OpenReview, 2025.
- [DynamicKV: Per-Layer Token Budgets](https://arxiv.org/html/2603.20397v1) — 2026.
- [LMCache: Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665) — 2025.
- [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079) — FAST'25.
- [KV Cache Compression Review](https://arxiv.org/pdf/2508.06297) — 2025.
- [Making Sense of KV Cache Optimizations](https://www.zansara.dev/posts/2025-10-26-kv-caching-optimizations-intro/) — Sara Zan, 2025.

### Attention Optimization

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/publications/flash3/flash3.pdf) — Tri Dao et al.
- [FlashAttention-3 — PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783) — 2024.
- [Token Sparse Attention: Efficient Long-Context Inference](https://arxiv.org/html/2602.03216) — 2026.
- [TokenRing: Efficient Parallelism via Bidirectional Communication](https://arxiv.org/html/2412.20501v1) — 2024.
- [Scaling LLM Inference: Innovations in TP, CP, and EP](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/) — Meta Engineering, 2025.

### Prefill-Decode Disaggregation

- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670) — OSDI'24.
- [Disaggregated Inference: 18 Months Later](https://haoailab.com/blogs/distserve-retro/) — Hao AI Lab, 2025.
- [DuetServe: Harmonizing Prefill and Decode](https://arxiv.org/pdf/2511.04791) — 2025.
- [SPAD: Specialized Prefill and Decode Hardware](https://augustning.com/assets/papers/spad-arxiv-2025.pdf) — 2025.
- [ShuffleInfer: Disaggregate LLM Inference for Mixed Workloads](https://dl.acm.org/doi/full/10.1145/3732941) — ACM TACO, 2025.
- [Prefill-Decode Disaggregation — BentoML Handbook](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)

### Quantization

- [GPTQ and AWQ Quantization on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/)
- [vLLM Quantization Guide with Benchmarks](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
- [NVIDIA Blackwell: The Impact of NVFP4 for LLM Inference](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/)
- [The New LLM Inference Stack 2025: FA-3, FP8 & FP4](https://www.stixor.com/blogs/new-inference-stack-2025)

### MoE Inference

- [Survey on Inference Optimization for Mixture of Experts Models](https://dl.acm.org/doi/10.1145/3794845) — ACM Computing Surveys, 2025.
- [Deploying DeepSeek with PD Disaggregation and Large-Scale EP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) — LMSYS, 2025.
- [DeepEP: Efficient Expert-Parallel Communication Library](https://github.com/deepseek-ai/DeepEP) — DeepSeek AI.
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — vLLM Blog.
- [MoE Inference on NVIDIA Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVIDIA, 2025.
- [Scaling DeepSeek-style MoEs with vLLM and llm-d](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep) — Red Hat, 2025.

### Structured Generation

- [XGrammar: Flexible and Efficient Structured Generation Engine](https://arxiv.org/pdf/2411.15100) — MLSys'25.
- [XGrammar-2: Efficient Dynamic Structured Generation for Agentic LLMs](https://arxiv.org/html/2601.04426v2) — 2026.

### Scheduling and Batching

- [FlowPrefill: Decoupling Preemption from Prefill Scheduling](https://arxiv.org/html/2602.16603) — 2026.
- [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115) — 2025.
- [Optimizing LLM Inference: Fluid-Guided Online Scheduling](https://arxiv.org/html/2504.11320v2) — 2025.
- [Helix: Serving LLMs over Heterogeneous GPUs via Max-Flow](https://dl.acm.org/doi/10.1145/3669940.3707215) — ASPLOS'25.

### Agentic Inference

- [Efficient LLM Serving for Agentic Workflows (Helium)](https://arxiv.org/html/2603.16104) — 2026.
- [Continuum: Efficient and Robust Multi-Turn LLM Agent](https://arxiv.org/pdf/2511.02230) — 2025.
- [Act While Thinking: Pattern-Aware Speculative Tool Execution](https://arxiv.org/html/2603.18897v1) — 2026.

### Network Optimization

- [NCCL 2.27 Release Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27) — NVIDIA.
- [Beyond NCCL: Faster Inter-Node All-Reduce for Decode](https://pssg.cs.umd.edu/blog/2025/beyond-nccl/) — UMD, 2025.
- [MSCCL++: Rethinking GPU Communication Abstractions](https://arxiv.org/html/2504.09014v1) — Microsoft, 2025.

### Memory and Hardware

- [Memory Bottleneck Emerges as Main LLM Inference Challenge](https://winbuzzer.com/2026/01/26/memory-bottleneck-llm-inference-hardware-challenge-xcxwbn/) — Winbuzzer, 2026.
- [Overcoming the AI Memory Wall with CXL Memory Pooling](https://computeexpresslink.org/blog/overcoming-the-ai-memory-wall-how-cxl-memory-pooling-powers-the-next-leap-in-scalable-ai-computing-4267/) — CXL Consortium.
- [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/html/2503.08311v2) — 2025.
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [AMD MI350 Series](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-game-changer.html)
- [ROCm 7.0 Software](https://www.amd.com/en/blogs/2025/rocm7-supercharging-ai-and-hpc-infrastructure.html)

### Diffusion LLMs

- [Fast-dLLM: Training-free Acceleration of Diffusion LLM](https://arxiv.org/pdf/2505.22618) — 2025.
- [ParallelBench: Trade-offs of Parallel Decoding in Diffusion LLMs](https://arxiv.org/abs/2510.04767) — 2025.
- [Learning to Parallel: Accelerating Diffusion LLMs (Learn2PD)](https://arxiv.org/abs/2509.25188) — 2025.

### Framework Comparisons and Benchmarks

- [vLLM vs TensorRT-LLM vs SGLang: H100 Benchmarks 2026](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/) — Spheron.
- [SGLang vs vLLM in 2026: Benchmarks and Architecture](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) — Particula.
- [vLLM vs SGLang vs LMDeploy: Fastest Inference Engine in 2026](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/) — PremAI.
- [vLLM Q1 2026 Roadmap](https://github.com/vllm-project/vllm/issues/32455)
- [vLLM Q4 2025 Roadmap](https://github.com/vllm-project/vllm/issues/26376)

### Serving Frameworks

- [NVIDIA Dynamo: Distributed Inference Framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — NVIDIA.
- [NVIDIA Dynamo 1.0 Production Ready](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) — NVIDIA.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) — llm-d.
- [NIXL: NVIDIA Inference Transfer Library](https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/) — NVIDIA.
- [SGLang Production Deployment Guide](https://www.spheron.network/blog/sglang-production-deployment-guide/) — Spheron, 2026.
