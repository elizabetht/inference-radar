# First-Principles Analysis of LLM Inference: Elon Musk's 5-Step Design Process Applied

*A companion to [inference-optimization-deep-dive.md](./inference-optimization-deep-dive.md)*

> **DGX Spark adaptation note**: This document was adapted from sara4dev/inference-research (March 2026).
> Hardware context for this lab: 2x GB10 Grace Blackwell (SM121, arm64, 128GB unified memory),
> PP=2 across spark-01/spark-02, model: Qwen/Qwen3-Coder-Next-FP8.
> Key GB10-specific callouts are annotated with **[GB10]** inline.

**Date**: March 2026
**Methodology**: Elon Musk's 5-Step Design Process — (1) Make the requirements less dumb, (2) Delete the part or process, (3) Simplify or optimize, (4) Accelerate cycle time, (5) Automate — applied as a first-principles audit of the modern LLM inference stack.

---

## Table of Contents

1. [Step 1: Make the Requirements Less Dumb](#step-1-make-the-requirements-less-dumb)
2. [Step 2: Delete the Part or Process](#step-2-delete-the-part-or-process)
3. [Step 3: Simplify or Optimize](#step-3-simplify-or-optimize)
4. [Step 4: Accelerate Cycle Time](#step-4-accelerate-cycle-time)
5. [Step 5: Automate](#step-5-automate)
6. [The 10x Roadmap](#the-10x-roadmap)

---

## Step 1: Make the Requirements Less Dumb

The most dangerous requirements are the ones nobody questions because a smart person wrote them. Every assumption in the modern inference stack was a reasonable decision at one point — but the conditions under which those decisions were made have changed radically. Here we interrogate ten foundational assumptions.

---

### 1.1 Why Continuous Batching?

**Assumption challenged**: Continuous batching — assembling heterogeneous requests into a shared batch at every decode step — is the only way to keep GPUs busy.

**Who decided this**: The [Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) introduced iteration-level scheduling. vLLM, SGLang, TensorRT-LLM, and every major framework adopted it as gospel. Before Orca, static batching wasted 60-80% of GPU cycles waiting for the longest sequence in a batch to finish.

**First principles reasoning**: The real requirement is *maximizing arithmetic intensity* — the ratio of FLOPs to bytes moved. Continuous batching achieves this by packing more tokens into each matrix multiplication. But it introduces substantial CPU-side scheduling overhead. Recent profiling shows [over 15% of GPU time is consumed by microsecond-scale bubbles](https://arxiv.org/html/2504.19516v1) caused by batch metadata updates, token transfers for streaming, and scheduling decisions between iterations. The GPU idle gap between decoding steps *increases with batch size* because the CPU must do more bookkeeping.

**What if instead**: Request-level parallelism — dedicating GPU resources to individual requests — eliminates scheduling overhead entirely. The [Parallel Track (PT) Transformer](https://arxiv.org/html/2602.07306) from February 2026 divides the model into several smaller independent "tracks" rather than partitioning parameters within layers. This achieves up to a **16x reduction in synchronization operations** relative to standard tensor parallelism. When integrated into TensorRT-LLM and vLLM, PT reports 15-30% reduced TTFT, 2-12% reduced TPOT, and up to 31.9% increased throughput.

**Evidence**: The [Bullet system](https://arxiv.org/html/2504.19516v1) (2025) demonstrates that dynamic spatial-temporal orchestration — filling GPU bubbles with useful work from other requests — outperforms pure continuous batching. [Hummingbird](https://arxiv.org/pdf/2601.04071) achieves 82% GPU utilization vs. 67% baseline through microsecond-scale preemption.

**Potential impact**: 15-30% throughput improvement by eliminating scheduling bubbles; potentially 2-3x with full request-level parallelism on future hardware with better isolation primitives.

**Who should build this**: vLLM and SGLang should explore PT-style track parallelism as an alternative scheduling mode. NVIDIA Dynamo's disaggregated architecture is already partially moving this direction.

**Why hasn't it been done**: Continuous batching is a local optimum. It works well enough. The engineering effort to redesign schedulers around request-level parallelism is massive, and the gains are "only" 15-30% — not enough to justify a rewrite when there are easier wins. But as those easy wins get exhausted, this becomes the next frontier.

---

### 1.2 Why KV Cache Per-Layer?

**Assumption challenged**: Every transformer layer needs its own unique KV cache entries.

**Who decided this**: The original transformer architecture (Vaswani et al., 2017). Each layer computes unique Q, K, V projections, so each layer's KV cache is stored independently. For a 70B parameter model at 128K context, this means ~40GB of KV cache alone.

**First principles reasoning**: Information theory tells us that if two signals are highly correlated, storing both is wasteful. Research consistently shows that KV representations across adjacent layers are *remarkably similar*, especially in the deeper layers of the network. This makes sense: residual connections mean each layer's output is a small perturbation of its input. By layer 20 of a 32-layer model, the hidden states have converged significantly.

**Evidence**:

- [**Cross-Layer Attention (CLA)**](https://www.marktechpost.com/2024/05/25/mit-researchers-propose-cross-layer-attention-cla/) from MIT shares KV activations across adjacent layers, achieving **2x KV cache reduction** with less than 1% perplexity degradation.
- [**KVSharer**](https://openreview.net/forum?id=2Akf4BBCKo) (2024) achieves **30% KV cache compression** as a plug-and-play method — with the counterintuitive finding that sharing *dissimilar* KV caches preserves quality better than sharing similar ones.
- [**MiniCache**](https://www.emergentmind.com/topics/efficient-kv-caching) (2024) merges adjacent layers' KV states, achieving up to **5x compression** (41% less memory) with near-lossless performance.
- [**DeepSeek's MLA**](https://arxiv.org/abs/2405.04434) compresses KV into a low-rank latent space, achieving a **93.3% reduction** in KV cache size and **5.76x** generation throughput improvement.

**Potential impact**: 2-5x memory reduction in KV cache, directly translating to 2-5x larger batch sizes or longer contexts on the same hardware. DeepSeek-V2 demonstrated 5.76x throughput improvement from MLA alone.

**Who should build this**: This should be a training-time architectural decision. New models should adopt MLA or CLA by default. For existing models, KVSharer and MiniCache offer post-training solutions.

**Why hasn't it been done universally**: MLA requires retraining from scratch. CLA and KVSharer are post-hoc and lose some quality. The industry inertia around standard multi-head attention is enormous — every kernel, every optimization, every piece of infrastructure assumes per-layer KV. DeepSeek broke this mold, and others (Kimi K2, GLM-5, Mistral 3 Large) are following.

---

### 1.3 Why the Same Precision Everywhere?

**Assumption challenged**: All layers should use the same numerical precision during inference.

**Who decided this**: Convenience. Uniform FP16 or BF16 is simple to implement, simple to reason about, and "good enough." Even quantization schemes like GPTQ and AWQ typically apply the same bit-width uniformly across all layers.

**First principles reasoning**: Different layers have fundamentally different roles and sensitivities. Embedding layers and the final language model head are notoriously sensitive to quantization — they map between a discrete vocabulary and a continuous representation space. Middle layers performing pattern matching are far more robust. [Research on mixed INT4-INT8 quantization](https://www.researchgate.net/publication/393117755) confirms that layer-specific sensitivity varies by orders of magnitude.

**Evidence**:

- [Amdahl's Law analysis for LLMs](https://openreview.net/forum?id=JtrQJJQYpP) shows that in large models like LLaMA and OPT, **projection layers account for more than 95% of total latency**. Quantizing these aggressively yields substantial benefits, while attention layers contribute negligibly to latency.
- [ATOM](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) (MLSys 2024) demonstrates mixed-precision quantization achieving near-FP16 quality at INT4 speeds by protecting sensitive channels.
- Deploying Gemma 3 27B in BF16 requires 54 GB; INT4 requires only 14.1 GB — a **4x reduction**. But uniform INT4 degrades quality on sensitive layers.
- NVIDIA's NVFP4 format in TensorRT-LLM and the FP8 support across all major frameworks demonstrate industry movement toward lower precision, but still mostly uniformly applied.

**Potential impact**: 2-4x memory reduction with less quality loss than uniform quantization. Per-layer mixed precision could enable running 70B models on single GPUs that currently require 2-4 GPUs.

**Who should build this**: Quantization toolkits (AutoGPTQ, AWQ, llama.cpp) should default to sensitivity-aware mixed precision. TensorRT-LLM's NVFP4 and FP8 support is a step in this direction.

**Why hasn't it been done**: The calibration process is model-specific and time-consuming. You need representative data to measure per-layer sensitivity. The tooling for automated sensitivity analysis is immature. And hardware support for truly mixed-precision execution (different layers at different precisions within one forward pass) is still emerging.

---

### 1.4 Why Transfer Full KV Cache in Disaggregated Serving?

**Assumption challenged**: When disaggregating prefill and decode onto separate GPUs, we must transfer the complete KV cache between them.

**Who decided this**: The natural architecture of disaggregated serving. NVIDIA Dynamo, Mooncake, and DistServe all separate prefill (compute-intensive) from decode (memory-bandwidth-intensive). The KV cache computed during prefill must be available for decode. The obvious approach: send it all.

**First principles reasoning**: The KV cache contains massive redundancy. For a 4K token prefill on Llama-70B, the KV cache is ~1.3GB. Transferring this over NVLink (900 GB/s) takes ~1.4ms, but over network (100 Gbps) takes ~100ms — dominating the entire TTFT. Information theory says we should only transfer the *information content*, not the raw representation.

**Evidence**:

- [**HACK**](https://arxiv.org/html/2502.03589v1) (SIGCOMM 2025) applies homomorphic compression to KV cache transfer, reducing job completion time by **up to 70.9%** compared to baseline disaggregated inference.
- [**Delta encoding for KV cache**](https://github.com/cenconq25/delta-compress-llm) exploits temporal coherence, achieving **F16-quality at Q4 compression ratios with zero perplexity loss** — no learned components, no entropy coding, no extra parameters.
- [**CacheGen**](https://dl.acm.org/doi/10.1145/3651890.3672274) (SIGCOMM 2024) streams compressed KV cache with adaptive coding, achieving significant bandwidth reduction.
- [**TraCT**](https://arxiv.org/html/2512.18194v1) uses CXL shared memory as a KV-transfer substrate, eliminating the NIC hop entirely — GPUs write and read KV blocks directly through CXL load/store.

**Potential impact**: 4-8x reduction in KV cache transfer size, reducing disaggregated serving TTFT from network-bound ~100ms to ~15-25ms. Combined with CXL-based shared memory, transfer could become near-zero cost.

**Who should build this**: NVIDIA Dynamo and llm-d should integrate compressed KV transfer as a default. LMCache already supports hierarchical storage with compression.

**Why hasn't it been done**: Disaggregated serving itself is new (production only since 2025). Teams are still getting basic disaggregation working. Compression adds complexity and potential failure modes. But the HACK paper showing 70.9% JCT reduction makes this a clear next step.

---

### 1.5 Why Does Every Token Get the Same Compute?

**Assumption challenged**: Every token — whether "the", "is", or "quantum entanglement" — requires a full forward pass through all transformer layers.

**Who decided this**: The autoregressive generation loop. Each token is produced by running the full model. This treats the prediction of "the" (which follows "in" with >30% probability in English) identically to predicting a rare technical term.

**First principles reasoning**: The entropy of the next-token distribution varies enormously. For high-confidence predictions (low entropy), most of the model's computation is wasted — the answer was already determined by the first few layers. Information theory says we should allocate compute proportional to the information content of each decision.

**Evidence**:

- [**LayerSkip**](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/) (Meta, 2024) enables early exit at earlier layers, achieving **1.5-2.5x speedup** with negligible quality loss. It applies layer dropout during training with increasing rates for later layers.
- [**SWIFT**](https://openreview.net/forum?id=EKJhH5D5wA) (2024) adaptively selects which layers to skip per-token, achieving **1.3-1.6x speedup** while preserving the original distribution.
- [**FlexiDepth**](https://arxiv.org/html/2503.23798) (2025) determines at each layer whether the hidden state should pass through or skip, offering per-token computation paths.
- [**Mixture-of-Depths (MoD)**](https://arxiv.org/html/2503.23798) injects a router at every layer to skip unimportant layers per-token.
- [**EAGLE-3**](https://arxiv.org/html/2503.01840v1) (NeurIPS 2025) takes the opposite approach: use a tiny draft head (<5% extra parameters) to predict multiple tokens speculatively, then verify in one pass. Achieves **2-6x speedup** with identical output quality. Acceptance rate of 70-80% across positions.

**Potential impact**: 2-6x speedup for generation-heavy workloads. EAGLE-3 already demonstrates this in production. Combining speculative decoding with early exit could compound the gains.

**Who should build this**: EAGLE-3 is already integrated into vLLM and TensorRT-LLM. LayerSkip requires training-time modifications. The missing piece is a unified framework that combines speculative decoding, early exit, and adaptive depth at the serving level.

**Why hasn't it been done universally**: Speculative decoding (the most production-ready approach) requires a draft model or head, adding system complexity. Early exit requires retraining. The gains are model-dependent and hard to predict without profiling. But EAGLE-3's results make this a clear win for any deployment.

---

### 1.6 Why Fixed Batch Sizes for Prefill?

**Assumption challenged**: Prefill processes the entire prompt in one or a few large chunks before any decode begins.

**Who decided this**: Early inference systems treated prefill as a monolithic operation. You tokenize the prompt, run it through the model in one forward pass, then start generating. This creates a binary phase distinction.

**First principles reasoning**: The prefill/decode distinction is an artifact of the autoregressive architecture, not a physical law. Prefill is compute-bound; decode is memory-bound. Running them separately wastes resources — during prefill, memory bandwidth sits idle; during decode, compute sits idle. What if we interleaved them?

**Evidence**:

- [**Chunked prefill**](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a) breaks prefill into smaller chunks that can be batched with ongoing decode operations. Empirical evidence shows **+50% throughput gain** (TNG, 2025).
- [**POD-Attention**](https://akkamath.github.io/files/ASPLOS25_POD.pdf) (ASPLOS 2025) achieves full prefill-decode overlap, computing attention **up to 59% faster** (mean 28%) than FlashAttention and FlashInfer's separate kernels.
- [**TaiChi**](https://arxiv.org/html/2508.01989v1) dynamically switches between aggregated and disaggregated PD modes for goodput optimization.
- [**Nexus**](https://arxiv.org/html/2508.01989v1) decouples at the batching level while preserving compatibility with existing attention mechanisms.

**Potential impact**: 28-59% improvement in attention computation (POD-Attention), 50% throughput improvement from chunked prefill. Combined, this could yield 2x overall throughput.

**Who should build this**: All major frameworks already support chunked prefill. The next step is full PD overlap (POD-Attention) and dynamic aggregation/disaggregation (TaiChi).

**Why hasn't it been done**: Chunked prefill *has* been done and is widely adopted. The frontier is full overlap and dynamic mode switching, which requires sophisticated kernel design and scheduling logic.

---

### 1.7 Why Do We Tokenize/Detokenize?

**Assumption challenged**: Text must be converted to subword tokens before the model can process it.

**Who decided this**: BPE tokenization (Sennrich et al., 2016) and its variants. Subword tokenization was a pragmatic solution to vocabulary size vs. sequence length tradeoffs. It reduces sequence length by ~3.8x compared to byte-level input.

**First principles reasoning**: Tokenization is a lossy compression of the input signal. It introduces artifacts: the same word tokenized differently depending on context, out-of-vocabulary handling, cross-lingual inconsistencies, and a fixed vocabulary that cannot adapt to new domains. The tokenizer itself adds latency (typically 1-5ms for short prompts, 10-50ms for long ones). More fundamentally, tokenization forces the model to learn the tokenizer's arbitrary segmentation decisions rather than working with the raw signal.

**Evidence**:

- [**ByT5**](https://gregrobison.medium.com/a-comparative-analysis-of-byte-level-and-token-level-transformer-models-in-natural-language-9fb4331b6acc) demonstrated byte-level processing is viable but 3.8x more expensive in sequence length.
- [**MrT5 (MergeT5)**](https://openreview.net/forum?id=VYWBMq1L7H) (ICLR 2025) integrates learned token deletion, **reducing byte-level sequence lengths by up to 75%** while maintaining ByT5-comparable accuracy.
- [**Byte Latent Transformer (BLT)**](https://aclanthology.org/2025.acl-long.453.pdf) uses dynamic boundary prediction based on byte entropies, segmenting into variable-size patches. Scales more efficiently than subword LLMs.
- [**EvaByte**](https://hkunlp.github.io/blog/2025/evabyte/) rivals tokenizer-based LMs despite using **5x less training data**, excels in coding tasks, and **decodes up to 2x faster**.
- [**SpaceByte**](https://proceedings.neurips.cc/paper_files/paper/2024/file/e1f418450107c4a0ddc16d008d131573-Paper-Conference.pdf) (NeurIPS 2024) significantly outperforms standard byte-level Transformers and MegaByte.

**Potential impact**: Eliminating tokenization overhead (1-50ms per request), eliminating an entire preprocessing/postprocessing step, and potentially improving quality on non-English languages and code. The 2x decode speedup from EvaByte is particularly notable.

**Who should build this**: This is a model architecture decision. New model families should seriously consider byte-level or dynamic-patch approaches. BLT and EvaByte show the path.

**Why hasn't it been done**: The 3.8x sequence length penalty was prohibitive for standard transformers. But BLT, MrT5, and EvaByte show that with dynamic patching and learned compression, the overhead can be reduced to near-zero. The real blocker is that the entire ecosystem (fine-tuning, evaluation, deployment) assumes tokenized input.

---

### 1.8 Why Synchronous Tensor Parallelism?

**Assumption challenged**: In tensor parallelism, all GPUs must synchronize (all-reduce) after every layer's attention and FFN computations.

**Who decided this**: The Megatron-LM paper (Shoeybi et al., 2019). Standard tensor parallelism splits weight matrices across GPUs, requiring all-reduce after each matrix multiply to recombine partial results. For a 32-layer model with TP=8, this means 128+ all-reduce operations per forward pass.

**First principles reasoning**: All-reduce is a *synchronization barrier*. Every GPU must wait for the slowest one. In an 8-GPU collective, if a single CPU core is delayed by 1ms due to OS scheduling, all other GPUs busy-wait, [amplifying a small per-core delay into a cluster-wide stall](https://arxiv.org/html/2603.22774v1). The theoretical overhead of all-reduce is O(2*(n-1)/n * data_size / bandwidth), but the practical overhead includes synchronization jitter, PCIe/NVLink contention, and NCCL library overhead.

**Evidence**:

- [**Parallel Track (PT) Transformer**](https://arxiv.org/html/2602.07306) (February 2026) divides the model into independent tracks, achieving **16x reduction in synchronization operations** with competitive quality.
- [**Helix Parallelism**](https://research.nvidia.com/publication/2025-07_helix-parallelism-rethinking-sharding-strategies-interactive-multi-million) (NVIDIA, July 2025) decouples attention and FFN sharding strategies — KV parallelism for attention, TP x EP for FFN — with a lightweight temporal pipeline. Achieves **1.5x TTL reduction** and supports **32x larger batches** under the same latency budget for DeepSeek-R1.
- Meta's [scaling LLM inference blog](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/) documents how combining TP, CP, and EP requires careful balance to avoid synchronization bottlenecks.

**Potential impact**: 15-30% latency reduction from PT Transformer. 1.5x from Helix Parallelism. Combined with async communication overlapping, potentially 2x throughput.

**Who should build this**: NVIDIA (Helix is theirs), vLLM, SGLang. The PT Transformer approach could be integrated into any framework.

**Why hasn't it been done**: Standard TP is deeply embedded in model architectures and NCCL communication patterns. Helix and PT require rethinking how models are sharded, which touches every layer of the stack. NVIDIA is the natural leader here given their hardware-software co-design advantage.

---

### 1.9 Why Square Attention Matrices?

**Assumption challenged**: We compute the full N x N attention matrix where N is the sequence length, even though most attention weights are near-zero after softmax.

**Who decided this**: The original transformer's scaled dot-product attention computes all pairwise interactions. FlashAttention (Dao et al., 2022) made this efficient by avoiding materializing the full matrix, but still computes all entries.

**First principles reasoning**: Softmax is a *sparsifying* operation. For typical attention distributions, the vast majority of entries are near-zero — the model attends to a small subset of positions. Computing all N^2 interactions is like multiplying a matrix by zero in most entries. For a 128K context, this means ~16 billion attention computations per layer, of which perhaps 50-90% contribute negligibly to the output.

**Evidence**:

- [**SpargeAttn**](https://arxiv.org/html/2502.18137v4) (2025) uses a two-stage online filter to predict sparse blocks, **skipping ~50% of computation** with exact or near-exact results.
- [**Block-Sparse FlashAttention (BSFA)**](https://arxiv.org/abs/2512.07011) computes exact query-key similarities to select top-k value blocks, pruning ~50% of blocks.
- [**Native Sparse Attention (NSA)**](https://arxiv.org/html/2502.18137v4) achieves **up to 9.0x forward and 6.0x backward speedup** at 64k context.
- [**Sliding window attention**](https://arxiv.org/html/2502.18137v4) (used in Mistral, Gemma) limits attention to a local window for most layers, with full attention only in select layers.

**Potential impact**: 2-9x speedup for long-context inference. For 128K+ contexts, sparse attention is not optional — it is a requirement.

**Who should build this**: SpargeAttn and NSA are already available. TensorRT-LLM has sparse attention guides. The gap is making sparse attention the *default* rather than an opt-in optimization.

**Why hasn't it been done universally**: Sparse attention patterns are model-specific and input-dependent. A generic sparse attention system that works across all models without quality loss is hard. But for long-context workloads, the 50%+ computation savings make this essential.

---

### 1.10 Why Same Model for All Queries?

**Assumption challenged**: Every user query, regardless of difficulty, is processed by the same (usually largest available) model.

**Who decided this**: Deployment simplicity. Running one model is easier than running a routing system plus multiple models.

**First principles reasoning**: The computational difficulty of language tasks varies by orders of magnitude. "What is 2+2?" should not require the same computation as "Explain the implications of Godel's incompleteness theorems for artificial intelligence." Running a 70B model for trivial queries is like using a nuclear reactor to charge a phone.

**Evidence**:

- [**RouteLLM**](https://github.com/Not-Diamond/awesome-ai-model-routing) routes queries between small and large models based on predicted difficulty, approaching Pareto-optimal cost-accuracy tradeoffs.
- [**MixLLM**](https://aclanthology.org/2025.naacl-long.545.pdf) (NAACL 2025) implements dynamic routing across multiple LLMs.
- [**RouterDC**](https://proceedings.neurips.cc/paper_files/paper/2024/file/7a641b8ec86162fc875fb9f6456a542f-Paper-Conference.pdf) (NeurIPS 2024) uses dual contrastive learning for query-based routing.
- Industry deployments show 40-60% of queries can be handled by models 10-50x smaller than the "flagship" model with comparable user satisfaction.

**Potential impact**: 3-10x cost reduction with minimal quality degradation on easy queries. For high-traffic deployments, this is the single largest cost optimization available.

**Who should build this**: Cloud providers and API gateway companies. Not-Diamond, Martian, and Unify already offer routing. The gap is integrating routing natively into inference frameworks.

**Why hasn't it been done universally**: Routing adds latency (the router itself must make a decision), requires maintaining multiple models (operational complexity), and has an error rate (misrouting a hard query to a small model produces poor results). But the 3-10x cost savings make this increasingly compelling.

---

## Step 2: Delete the Part or Process

Musk's rule: "If you're not occasionally adding things back, you're not deleting enough." Here we identify components of the inference pipeline that can be *eliminated entirely*.

---

### 2.1 Delete Redundant Attention Computation

**What to delete**: Full attention computation across all layers for all tokens.

**Evidence for deletion**:
- [HARP](https://arxiv.org/pdf/2507.01900) prunes **all attention heads in the top layers** with adaptive output rescaling, based on the observation that upper layers exhibit representational over-smoothing and increased head redundancy.
- [Research on the "BOS-sink" phenomenon](https://arxiv.org/html/2602.02195) shows many "collapsed" heads are not performing useful computation.
- LayerSkip and SWIFT demonstrate that **30-50% of layers can be skipped** for many tokens without quality degradation.

**Deletion strategy**: For a 32-layer model, a combination of (a) early exit after layer 20-24 for high-confidence tokens, (b) head pruning in layers 25-32, and (c) sparse attention in all layers could eliminate **50-70% of total attention FLOPs**.

**Risk**: Catastrophic failure on edge cases where deep-layer computation is critical. Mitigated by speculative verification (EAGLE-3 style).

---

### 2.2 Delete KV Cache for Some Layers

**What to delete**: Independent KV cache storage for redundant layers.

**Evidence for deletion**:
- Cross-Layer Attention: 2x cache reduction, <1% quality loss.
- MiniCache: 5x compression on deep-layer KV states.
- DeepSeek MLA: 93.3% KV cache reduction.
- [Entropy-guided KV caching](https://www.mdpi.com/2227-7390/13/15/2366) shows that many cached entries are never meaningfully attended to.

**Deletion strategy**: For a 32-layer model using CLA with sharing factor 2, only 16 layers need unique KV entries. Combined with MLA-style compression, the KV cache could shrink by **8-10x** from current baselines.

---

### 2.3 Delete Synchronization Barriers

**What to delete**: All-reduce synchronization after every layer in tensor parallelism.

**Evidence for deletion**:
- PT Transformer: 16x fewer synchronization operations with competitive quality.
- Helix Parallelism: decouples attention and FFN sharding, using lightweight temporal pipelining instead of full synchronization.
- In an 8-GPU setup, each all-reduce adds ~10-50 microseconds. With 64+ all-reduces per forward pass, this accumulates to **0.6-3.2ms of pure synchronization overhead** per token.

**Deletion strategy**: Replace per-layer all-reduce with (a) track-based independent computation (PT) or (b) phase-specific parallelism (Helix). For MoE models, expert parallelism already reduces synchronization by computing experts independently.

---

### 2.4 Delete the Prefill/Decode Distinction

**What to delete**: The hard boundary between prefill and decode phases.

**Evidence for deletion**:
- Chunked prefill already blurs this line, yielding +50% throughput.
- POD-Attention shows full overlap is possible, with 28-59% faster attention.
- TaiChi demonstrates that the optimal PD strategy is *dynamic*, not static.

**Deletion strategy**: Treat the entire inference pipeline as a continuous stream of mixed compute-bound and memory-bound operations. Schedule them on a unified kernel that dynamically allocates GPU resources based on the current mix. This eliminates the concept of "phases" entirely.

---

### 2.5 Delete Tokenization Overhead

**What to delete**: The tokenizer/detokenizer preprocessing step.

**Evidence for deletion**:
- EvaByte rivals tokenizer-based LMs with 5x less training data and decodes 2x faster.
- BLT's dynamic patching achieves better scaling than subword models.
- MrT5 reduces byte-level overhead by 75% through learned merging.

**Deletion strategy**: For new model architectures, adopt byte-level processing with dynamic patching (BLT-style). For existing models, this is not deletable without retraining — but the tokenization step itself (typically 1-50ms) can be overlapped with other operations.

**Caveat**: This is a training-time decision. You cannot delete the tokenizer from an already-trained model.

---

### 2.6 Delete Redundant Expert Computation in MoE

**What to delete**: Loading and computing experts that will not be selected by the router.

**Evidence for deletion**:
- In DeepSeek-V3 (256 experts), only **8 experts** are activated per token. Loading all 256 into GPU memory or computing router scores for all is wasteful.
- [Expert prediction](https://dl.acm.org/doi/10.1145/3794845) based on the previous layer's gating inputs achieves **~90% prediction accuracy** due to residual structure similarity.
- [ExpertCache](https://conf.researchr.org/details/icsme-2025/icsme-2025-nier/17/ExpertCache) uses RL to optimize which experts to cache in GPU memory and which to activate, eliminating unnecessary loads.
- Pre-gated MoE identifies required experts during the current layer's computation and preloads them, overlapping transfer with computation.

**Deletion strategy**: Predict next-layer experts during current-layer computation (~90% accuracy). Preload predicted experts while current layer computes. For the ~10% mispredictions, use a fallback mechanism (compute with cached approximate expert or re-route).

**Potential impact**: For offloaded MoE serving (where experts live on CPU/NVMe), this eliminates **~90% of expert load latency**. For fully GPU-resident serving, it enables fitting larger MoE models by caching only frequently-used experts.

---

## Step 3: Simplify or Optimize

Only after questioning every requirement and deleting what we can do we optimize what remains. Here we analyze the fundamental performance bounds.

---

### 3.1 Theoretical Minimum Compute Per Token

**The math**: For a transformer with L layers, hidden dimension d, and vocabulary size V:
- Minimum FLOPs per token (decode) = 2 * P, where P is the parameter count (each parameter is multiplied once and accumulated once)
- For Llama-70B: ~140 GFLOPs per token
- For Llama-8B: ~16 GFLOPs per token

**Current reality on H100** (989 TFLOPS FP16, 3.35 TB/s HBM bandwidth):
- Compute-limited throughput: 989 TFLOPS / 140 GFLOPs = ~7,064 tokens/sec for 70B
- Bandwidth-limited throughput: 3.35 TB/s / 140 GB (model at FP16) = ~24 tokens/sec for 70B
- **The decode phase is ~300x more bandwidth-limited than compute-limited**

This means the GPU's compute units are **~99% idle** during decode for batch size 1. This is the fundamental insight: decode-phase inference is not a compute problem, it is a memory bandwidth problem.

**How far are current systems from theoretical limits**:

| Metric | Theoretical Max | Current Best | Gap |
|--------|----------------|--------------|-----|
| Decode tokens/sec (70B, BS=1, H100) | ~24 tok/s | ~18 tok/s | 1.3x |
| Decode tokens/sec (70B, BS=256, H100) | ~6,100 tok/s | ~2,500 tok/s | 2.4x |
| Prefill tokens/sec (70B, H100) | ~7,000 tok/s | ~3,500 tok/s | 2.0x |

The gap narrows at batch size 1 (we are already close to the bandwidth wall) but widens at high batch sizes where scheduling overhead, memory management, and synchronization waste significant resources.

---

### 3.2 Memory Bandwidth Utilization

**The bottleneck hierarchy** (from [Efficient LLM Inference](https://arxiv.org/html/2507.14397v1)):

1. **Memory bandwidth** (the wall): HBM3e on H100 delivers 3.35 TB/s. Current systems achieve 70-85% utilization during decode. The remaining 15-30% is lost to non-contiguous memory access patterns (paged KV cache) and cache hierarchy inefficiencies.

2. **Synchronization latency** (the amplifier): Sub-microsecond all-reduce across 64-128 chips is essential. Latencies exceeding 2.5 microseconds severely degrade performance gains from increased bandwidth. Current NVLink latencies are 1-5 microseconds.

3. **Compute capacity** (rarely the bottleneck): Tensor utilization remains less than or equal to 1% in low-batch decode scenarios. Even at high batch sizes, compute headroom exists.

4. **Memory capacity** (the constraint): For Llama-405B at 64K context with 32 concurrent users, ~881GB total capacity is needed. This determines how many requests can be served concurrently.

**Performance bounds** from the same paper: Current technology reaches ~2,000 user tokens per second on large models at reasonable context lengths. Getting to 10,000+ tokens/sec requires "smaller models, smaller context, or other forms of algorithmic advances" — hardware alone cannot bridge the gap.

**The B200 improvement**: 192 GB HBM3e at 8.0 TB/s bandwidth — a 2.4x bandwidth increase over H100. This directly translates to ~2.4x decode throughput improvement for bandwidth-bound workloads, which is exactly what B200 benchmarks show.

---

### 3.3 GPU Utilization: Where Are the Bubbles?

**Profiling results** from [recent systematic characterization](https://arxiv.org/html/2512.01644v1):

- **15%+ of GPU time** is consumed by microsecond-scale bubbles during inference (Llama-8B, DeepSeek-16B)
- Bubbles originate from: iteration-level device-host memory operations, synchronizations, transferring generated tokens for streaming, updating batch metadata for continuous batching
- **Average DRAM and compute utilization remains below 50%** at large batch sizes despite peak utilization approaching saturation
- GPU idle gap between decode steps increases with batch size due to CPU processing time

**The CPU bottleneck nobody talks about**: [Recent research](https://arxiv.org/html/2603.22774v1) shows that in multi-GPU inference, a single delayed CPU core can stall all GPUs via NCCL busy-wait. CPU-induced slowdowns of 1ms per core translate to cluster-wide stalls.

**Where time is actually spent** (approximate breakdown for decode on 8xH100):

| Component | Time % | Type |
|-----------|--------|------|
| HBM reads (model weights) | 45-55% | Memory bandwidth |
| HBM reads/writes (KV cache) | 15-25% | Memory bandwidth |
| Compute (matmul, attention) | 5-15% | Compute |
| All-reduce communication | 5-10% | Synchronization |
| Scheduling + metadata | 5-10% | CPU overhead |
| Kernel launch + misc | 3-5% | System overhead |

The biggest Amdahl's Law bottleneck is HBM bandwidth for model weight reads. No amount of optimization to other components will help if you cannot read the weights faster. This is why quantization (reducing weight size) and batching (amortizing weight reads across more tokens) are the two most impactful optimizations.

---

### 3.4 The Bandwidth Wall: First Principles

**The physics**: HBM bandwidth is limited by the number of memory channels, the clock rate, and the bus width. HBM3e on H100 uses a 1024-bit bus per stack with 8 channels. Increasing bandwidth requires:
- More stacks (limited by package area and power)
- Higher clock rates (limited by signal integrity and power)
- Wider buses (limited by physical routing)

NVIDIA's trajectory: H100 (3.35 TB/s) -> H200 (4.8 TB/s) -> B200 (8.0 TB/s). That is a ~2.4x improvement per generation, roughly matching Moore's Law. But model sizes are growing faster.

**The implication**: For bandwidth-bound decode, throughput improvement tracks HBM bandwidth improvement — about 2-2.5x per GPU generation. To achieve 10x improvement, we need algorithmic changes:
1. **Reduce what must be read**: Quantization (4x from FP16 to INT4), weight sharing, sparse models
2. **Read less often**: Larger batch sizes (amortize weight reads), speculative decoding (generate multiple tokens per read)
3. **Read faster**: Multi-GPU parallelism (aggregate bandwidth), CXL-attached memory pools, processing-in-memory

The theoretical 10x decode speedup recipe: 4x from INT4 quantization x 2.5x from larger batch sizes = 10x. This is achievable today on hardware like B200 with proper optimization.

---

## Step 4: Accelerate Cycle Time

What is slowing down the *iteration speed* of inference optimization itself? What tools, infrastructure, and processes are missing?

---

### 4.1 Benchmarking Infrastructure Gaps

**What's missing**:

1. **No standard end-to-end latency breakdown benchmark**: MLPerf measures aggregate throughput and latency, but does not break down where time is spent (compute vs. memory vs. communication vs. scheduling). [TokenPowerBench](https://arxiv.org/html/2512.03024) is the first to couple phase-aware telemetry with token-level normalization, but it focuses on power, not latency decomposition.

2. **No production-realistic workload generator**: Existing benchmarks use synthetic Poisson arrivals with uniform prompt/completion lengths. Real workloads have bursty arrivals, bimodal length distributions, and correlated patterns. [GuideLLM](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference) begins to address this but is limited.

3. **No cross-framework apples-to-apples comparison**: Benchmarking vLLM vs. SGLang vs. TensorRT-LLM requires identical configurations, which are nearly impossible to achieve given different default settings, different quantization implementations, and different scheduling policies.

4. **No long-context inference benchmark**: Most benchmarks test at 2K-8K context. Production workloads increasingly use 32K-128K+. Performance characteristics change dramatically at long contexts (attention becomes quadratic, KV cache dominates memory).

---

### 4.2 Profiling Tools That Don't Exist

**Gaps identified** from [eInfer](https://dl.acm.org/doi/pdf/10.1145/3748355.3748372) and related work:

1. **No request-level distributed trace for inference**: Traditional GPU profiling (NVIDIA CUPTI, NSight) lacks CPU-side visibility and request-level context. Framework-specific profilers (PyTorch Profiler) are confined to their stack. There is no equivalent of Jaeger/Zipkin for inference requests that traces a request from API entry through scheduling, prefill, decode, and response streaming across multiple GPUs.

2. **No real-time Amdahl's Law bottleneck identifier**: A tool that continuously profiles running inference and shows "right now, 45% of your latency is HBM bandwidth, 25% is KV cache reads, 15% is all-reduce, 10% is scheduling, 5% is other" — and recommends optimizations accordingly.

3. **No automated regression detection for latency**: When a framework update or configuration change causes a 5% latency regression, there is no standard CI/CD pipeline that catches this. [Bench360](https://arxiv.org/pdf/2511.16682) proposes local LLM benchmarking from multiple angles but is not integrated into CI.

---

### 4.3 A/B Testing for Serving Configurations

**What's missing**: A framework for safely A/B testing inference configurations in production:
- Canary deployments with automatic rollback on latency regression
- Statistical significance testing for throughput/latency differences
- Traffic splitting by request type (short vs. long, easy vs. hard)
- Cost-normalized comparison (tokens per dollar, not just tokens per second)

**Who should build this**: Kubernetes-native platforms like llm-d are the natural home. With llm-d's v0.5 release adding active-active HA and autoscaling, the infrastructure foundation exists.

---

### 4.4 What Would Accelerate Iteration

1. **One-click reproducible benchmarks**: `llm-bench run --model llama-70b --framework vllm --hardware 8xH100 --workload production-realistic` that produces a standardized report
2. **Automatic configuration search**: Given a model and hardware, find the optimal TP/PP/EP/batch-size/quantization configuration in hours, not weeks
3. **Regression-aware CI/CD**: Every framework PR automatically benchmarked against a standard workload suite
4. **Shared profiling datasets**: Anonymized production traces that researchers can use to develop better schedulers

---

## Step 5: Automate

What is currently done by humans that should be automated, because humans are too slow and make too many errors?

---

### 5.1 Model-Specific Quantization Selection

**Current state**: Engineers manually choose between FP16, FP8, INT8, INT4, GPTQ, AWQ, SqueezeLLM, etc. They run a few benchmarks, eyeball the quality, and deploy.

**What should be automated**: Given a model and target hardware, automatically:
1. Profile per-layer sensitivity using calibration data
2. Select optimal per-layer precision (mixed INT4/INT8/FP8)
3. Validate quality on a standard eval suite
4. Report the Pareto frontier of quality vs. throughput vs. cost

**Who is working on this**: [AIConfigurator](https://arxiv.org/html/2601.06288v1) constructs a performance database through offline profiling and enables rapid configuration search without GPU-based profiling. [BentoML's LLM-Optimizer](https://www.bentoml.com/blog/announcing-llm-optimizer) automates benchmarking across parameter combinations.

**Gap**: Neither tool performs *automatic quality-aware* quantization selection. They optimize throughput/latency but leave quality validation to the user.

---

### 5.2 Optimal Parallelism Strategy Selection

**Current state**: Engineers choose TP=4 or TP=8 or PP=2,TP=4 based on rules of thumb and experience.

**What should be automated**: Given a model architecture, hardware topology (GPU count, NVLink topology, network bandwidth), and workload characteristics (average prompt length, batch size, latency SLO):
1. Enumerate feasible parallelism strategies (TP, PP, EP, CP, Helix, and combinations)
2. Predict performance of each using analytical models
3. Validate top candidates with short benchmark runs
4. Deploy the optimal configuration

**Evidence this is needed**: [The inference parallelism analysis](https://wilsonwu.me/en/blog/2025/llm-inference-parallelism-in-vllm/) shows that the optimal parallelism strategy depends on model size, sequence length, batch size, and hardware — a combinatorial space too large for manual exploration.

**Who should build this**: AIConfigurator is the closest, but it is limited to three frameworks. A universal parallelism optimizer integrated into llm-d or NVIDIA Dynamo would be transformative.

---

### 5.3 KV Cache Eviction Policy Tuning

**Current state**: Most frameworks use LRU (least recently used) or simple FIFO eviction for KV cache blocks. Some support prefix caching (SGLang's RadixAttention).

**What should be automated**:
- Workload-aware eviction: if requests share common system prompts, cache those; if requests are unique, evict eagerly
- Dynamic eviction threshold based on memory pressure and incoming request rate
- Predictive prefetching based on request patterns (e.g., if a user is having a multi-turn conversation, prefetch their KV cache before the next message arrives)

**Evidence**: SGLang's cache-aware load balancer predicts cache hit rates per worker and routes requests accordingly. This is a step toward automated eviction, but the eviction policy itself remains static.

---

### 5.4 Batch Size and Chunked Prefill Optimization

**Current state**: `max_num_batched_tokens` and chunk sizes are set as static configuration values. The optimal values depend on model size, hardware, workload mix, and latency SLOs.

**What should be automated**:
- Dynamic batch size that adapts to current load (larger batches at high load for throughput, smaller at low load for latency)
- Adaptive chunk size for chunked prefill based on the current mix of prefill vs. decode work
- Automatic detection of the "saturation batch size" — the point where adding more requests to a batch no longer improves throughput

**Who should build this**: All major frameworks. vLLM's documentation already mentions optimization tuning, but the values are static.

---

### 5.5 Speculative Decoding Draft Model Selection

**Current state**: Users manually choose a draft model for speculative decoding — if they use speculative decoding at all.

**What should be automated**:
- Automatic selection of the best draft mechanism (separate model, EAGLE head, Medusa heads, self-speculative) based on the target model and workload
- Dynamic adjustment of speculation depth based on acceptance rate (speculate more when acceptance is high, less when low)
- Online adaptation of the draft model to the current workload distribution (OSD - Online Speculative Decoding)

**Evidence**: [EAGLE-3](https://arxiv.org/html/2503.01840v1) achieves 70-80% acceptance rate and 2-6x speedup, but the optimal number of draft tokens varies by model and input. [TETRIS](https://aclanthology.org/2025.acl-long.1598.pdf) optimizes draft token selection at the batch level. Neither automates the selection of the speculative decoding *method*.

---

### 5.6 Autoscaling Policies

**Current state**: Kubernetes HPA (Horizontal Pod Autoscaler) scales based on CPU/memory/custom metrics. LLM-specific metrics (queue depth, TTFT SLO violations, batch utilization) require manual integration.

**What should be automated**:
- Scale-to-zero for idle models (llm-d v0.5 supports this)
- Predictive autoscaling based on traffic patterns (scale up before the morning rush, not during it)
- Cost-aware scaling that considers spot instance availability and pricing
- Quality-of-service-aware scaling that routes overflow to smaller/faster models rather than queueing

**Who should build this**: llm-d (now a CNCF sandbox project) and Kubernetes Gateway API Inference Extension are the natural homes. llm-d v0.5 already supports scale-to-zero and reproducible benchmark workflows.

---

## The 10x Roadmap

What would it take to achieve a 10x improvement in LLM inference efficiency?

### The Compounding Stack

These optimizations are *multiplicative*, not additive:

| Optimization | Speedup Factor | Readiness |
|-------------|---------------|-----------|
| INT4 mixed-precision quantization | 2-4x (memory) | Production-ready |
| Speculative decoding (EAGLE-3) | 2-3x (latency) | Production-ready |
| Sparse attention (long context) | 2-4x (attention) | Near-production |
| KV cache compression (MLA/CLA) | 2-5x (memory) | Requires retraining |
| Query routing (small model for easy queries) | 3-5x (cost) | Production-ready |
| Async parallelism (Helix/PT) | 1.3-1.5x (throughput) | Early production |
| Dynamic batch + prefill overlap | 1.5-2x (throughput) | Production-ready |

**Conservative 10x path** (no retraining required):
- INT4 quantization (2x memory) x EAGLE-3 (2x latency) x chunked prefill (1.5x throughput) x query routing (3x cost) = **18x cost efficiency improvement**

**Aggressive 10x path** (requires retraining):
- MLA (5x KV cache) x byte-level model (1.5x decode speed) x sparse attention (3x long-context) x INT4 (2x memory) = **45x efficiency improvement**

### The Real Blockers

1. **Ecosystem inertia**: Every optimization requires changes across the stack — from model architecture to kernels to scheduling to deployment. No single team controls the full stack.

2. **Quality evaluation lag**: We can measure throughput and latency instantly but measuring quality degradation from aggressive optimization requires extensive eval suites and human judgment.

3. **Hardware-software co-design gap**: The best optimizations (MLA, sparse attention, byte-level models) require hardware that is optimized for them. Current GPUs are designed for dense matrix multiplication.

4. **Lack of automated optimization**: Most optimizations are applied manually by expert engineers. The field needs an "auto-tune" that searches the optimization space automatically.

### What a 10x Inference System Looks Like (2027)

- **Model**: Byte-level MoE with MLA, trained with layer dropout for early exit, with native sparse attention
- **Serving**: Disaggregated prefill/decode with compressed KV transfer, Helix-style parallelism, dynamic aggregation/disaggregation
- **Scheduling**: Per-token adaptive compute depth, speculative decoding with online-adapted draft head, automated batch size tuning
- **Routing**: Difficulty-aware routing across model sizes, with automatic fallback
- **Hardware**: B200/B300 with 8+ TB/s HBM bandwidth, CXL-attached memory pools for KV cache sharing
- **Automation**: Fully automated quantization, parallelism selection, and autoscaling — no manual tuning required

---

## Sources

### Frameworks and Systems
- [vLLM Architecture](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [llm-d Architecture](https://llm-d.ai/docs/architecture)
- [llm-d CNCF Donation](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/)

### KV Cache Optimization
- [Cross-Layer Attention (CLA) - MIT](https://www.marktechpost.com/2024/05/25/mit-researchers-propose-cross-layer-attention-cla/)
- [KVSharer](https://openreview.net/forum?id=2Akf4BBCKo)
- [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434)
- [HACK: KV Cache Compression for Disaggregated Inference](https://arxiv.org/html/2502.03589v1)
- [Delta Encoding for KV Cache](https://github.com/cenconq25/delta-compress-llm)
- [LMCache](https://lmcache.ai/tech_report.pdf)
- [TraCT: CXL Shared Memory KV Cache](https://arxiv.org/html/2512.18194v1)

### Adaptive Compute and Speculative Decoding
- [LayerSkip - Meta](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [SWIFT](https://openreview.net/forum?id=EKJhH5D5wA)
- [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/html/2503.01840v1)
- [FlexiDepth](https://arxiv.org/html/2503.23798)
- [TETRIS](https://aclanthology.org/2025.acl-long.1598.pdf)

### Sparse Attention
- [SpargeAttn](https://arxiv.org/html/2502.18137v4)
- [Block-Sparse FlashAttention](https://arxiv.org/abs/2512.07011)
- [Native Sparse Attention (NSA)](https://arxiv.org/html/2502.18137v4)

### Parallelism
- [Parallel Track (PT) Transformer](https://arxiv.org/html/2602.07306)
- [Helix Parallelism - NVIDIA](https://research.nvidia.com/publication/2025-07_helix-parallelism-rethinking-sharding-strategies-interactive-multi-million)
- [Meta: Scaling LLM Inference](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)

### Byte-Level Models
- [Byte Latent Transformer (BLT)](https://aclanthology.org/2025.acl-long.453.pdf)
- [EvaByte](https://hkunlp.github.io/blog/2025/evabyte/)
- [MrT5 (ICLR 2025)](https://openreview.net/forum?id=VYWBMq1L7H)
- [SpaceByte (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/e1f418450107c4a0ddc16d008d131573-Paper-Conference.pdf)

### Quantization
- [Mixed-Precision Quantization Survey](https://arxiv.org/html/2510.16805v1)
- [ATOM (MLSys 2024)](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf)
- [Amdahl's Law for LLM Quantization](https://openreview.net/forum?id=JtrQJJQYpP)

### MoE Inference
- [MoE Inference Optimization Survey](https://dl.acm.org/doi/10.1145/3794845)
- [ExpertCache](https://conf.researchr.org/details/icsme-2025/icsme-2025-nier/17/ExpertCache)
- [Speculating Experts](https://arxiv.org/pdf/2603.19289)

### Performance Analysis
- [Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity](https://arxiv.org/html/2507.14397v1)
- [LLM Inference Roofline Model Survey](https://arxiv.org/html/2402.16363v5)
- [GPU Bottlenecks in Large-Batch Inference](https://arxiv.org/html/2503.08311v2)
- [CPU-Induced Slowdowns in Multi-GPU Inference](https://arxiv.org/html/2603.22774v1)
- [Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/html/2512.01644v1)

### Model Routing
- [MixLLM (NAACL 2025)](https://aclanthology.org/2025.naacl-long.545.pdf)
- [RouterDC (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/7a641b8ec86162fc875fb9f6456a542f-Paper-Conference.pdf)
- [Awesome AI Model Routing](https://github.com/Not-Diamond/awesome-ai-model-routing)

### Benchmarking and Profiling
- [GuideLLM](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference)
- [TokenPowerBench](https://arxiv.org/html/2512.03024)
- [eInfer](https://dl.acm.org/doi/pdf/10.1145/3748355.3748372)
- [AIConfigurator](https://arxiv.org/html/2601.06288v1)
- [BentoML LLM-Optimizer](https://www.bentoml.com/blog/announcing-llm-optimizer)

### Prefill/Decode Optimization
- [POD-Attention (ASPLOS 2025)](https://akkamath.github.io/files/ASPLOS25_POD.pdf)
- [TaiChi: Unified PD Aggregation/Disaggregation](https://arxiv.org/html/2508.01989v1)
- [Chunked Prefill Analysis](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)
