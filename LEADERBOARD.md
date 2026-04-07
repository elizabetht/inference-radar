# Inference Autoresearch Leaderboard

*Auto-generated after each experiment. Sorted by throughput (descending).*

| # | Experiment | Framework | Model | Node Config | tok/s | TTFT p50 (ms) | ITL p50 (ms) | Status | Date |
|--:|:-----------|:----------|:------|:------------|------:|--------------:|-------------:|:-------|:-----|
| — | vllm-prefix-caching-pp2 | vllm | Qwen3-Coder-Next-FP8 | pp2-spark01-spark02 | 0 | 0 | 0 | degraded (c=1 ok, c=8+ fail) | 2026-04-07 |

*1 experiment attempted. 0 clean results. Queue running.*

---

## Reference: sara4dev/inference-research Results (SM120, Qwen2.5-3B-Instruct)

*These results are from a different hardware setup (RTX PRO 6000, SM120, x86_64) and smaller model (3B BF16).*
*They inform experiment priorities — see `autoresearch/program.md` for adaptation notes.*

| # | Experiment | Framework | Model | GPUs | tok/s | TTFT p50 (ms) | ITL p50 (ms) |
|--:|:-----------|:----------|:------|-----:|------:|--------------:|-------------:|
| 1 | qwen-fp8-weights | vllm | Qwen2.5-3B-Instruct | 1 | 5863.5 | 28.8 | 3.9 |
| 2 | vllm-fp8-3b-prequant | vllm | Qwen2.5-3B-Instruct-FP8-dynamic | 1 | 5854.1 | 28.4 | 3.9 |
| 3 | sglang-combined-best | sglang | Qwen2.5-3B-Instruct | 1 | 4342.6 | 27.4 | 5.7 |
| 4 | sglang-baseline | sglang | Qwen2.5-3B-Instruct | 1 | 4291.5 | 28.0 | 5.7 |
| 5 | combined-best | vllm | Qwen2.5-3B-Instruct | 1 | 4286.8 | 33.7 | 5.7 |
| 6 | cuda-graphs-enabled | vllm | Qwen2.5-3B-Instruct | 1 | 4241.0 | 34.7 | 5.8 |
| 7 | trtllm-baseline | trtllm | Qwen2.5-3B-Instruct | 1 | 4390.1 | 41.2 | 5.6 |
| 8 | llmd-inference-scheduling | llmd | Qwen2.5-3B-Instruct | 1 | 4360.9 | 31.9 | 5.8 |
| 9 | baseline-ubuntu-desktop | vllm | Qwen2.5-3B-Instruct | 1 | 3553.0 | 38.7 | 6.6 |
| 10 | dynamo-disagg-3b-natsfix | dynamo | Qwen2.5-3B-Instruct | 2 | 3539.9 | 94.8 | 4.0 |

**Key learnings for this lab:**
- FP8 weights: +65% throughput (already in our baseline — Qwen3-Coder-Next-FP8)
- CUDA graphs: +19% (next to validate on GB10)
- SGLang > vLLM by ~21% at defaults on Blackwell
- TP>1 on Blackwell REQUIRES `--disable-custom-all-reduce`
- Dynamo disagg: worse TTFT (94ms vs 38ms), no throughput gain vs single-GPU at c≤32
- CUDA graphs HURT Dynamo disagg (-12% vs disagg eager) — do not combine
