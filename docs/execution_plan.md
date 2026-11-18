# Execution plan: compression-aware LLM serving platform

Living document. Update it whenever a decision changes. Last updated: 2026-11-03.

## 1. Goal

Serve an open LLM compressed with techniques I reimplement myself, on vLLM, deployed on Kubernetes with autoscaling and observability, and publish a benchmark write-up (latency / throughput / cost / quality vs. an unoptimized baseline) plus a live demo.

Why: close two profile gaps (no production LLM-serving experience; no GCP/Azure) and have a portfolio project that is clearly solo work.

## 2. Working agreement (how we build)

- Claude acts as build partner and teacher: explain the concept and tradeoffs briefly, then I make the design call where there is a real choice.
- I write the code with guidance. Claude writes full code only for boilerplate or when I ask directly.
- Claude periodically asks me to explain back what we built, interview-style.
- Claude flags where each piece maps to an ML system design interview question.
- New repo, no forks. Every compression technique is reimplemented cleanly by me, not copied from research code.

## 3. Decisions made

| # | Decision | Date |
|---|---|---|
| D1 | Main model: Llama 3.1 8B. A 1B-3B sibling is used for the fast dev loop. Qwen deferred (see D8). | 2026-10-24 |
| D2 | Pruning method: **Isomorphic Pruning** (structured). MaskLLM explicitly **not** used. | 2026-10-29 |
| D3 | KV compression: **training-free low-rank KV** (my research). Token **eviction** is a stretch goal, new to me. | 2026-10-29 |
| D4 | Quantization: reimplement GPTQ or AWQ-style weight-only quantization and export to a vLLM-loadable format. Exact method still to choose. | 2026-10-26 |
| D5 | Compute: **GCP only, no Delta** (revised 2026-11-18; previously: use 48 Delta hours first). Spot instances, stop VMs when idle. Code moves Mac -> private GitHub repo -> `git pull` on the GCP VM; weights/results stay on the VM disk or a bucket. If 24 GB L4 is too small for an experiment, request an A100 40 GB quota. | 2026-11-18 |
| D6 | Cloud for the K8s phase: **GCP (GKE)**, L4 24 GB GPU, spot node pools. Azure dropped. | 2026-11-03 |
| D7 | Experiment matrix: **Option A (lean)**: Baseline, Q, P, P+Q, L, Q+L on Llama 3.1 8B (6 variants). | 2026-11-03 |
| D8 | Eviction (E) and a Qwen run are stretch goals, only after Option A works end-to-end. | 2026-11-03 |
| D9 | Speed benchmarks run on ONE GPU type that I can always rent again (cloud L4 if GCP). Delta is used for offline work only (see section 5). | 2026-10-26 |
| D10 | Local dev on a MacBook Air M5 (24 GB, no CUDA): harness, compression code on a small model, K8s on kind with CPU or mock server. | 2026-10-24 |
| D12 | GCP region: **us-central1** (L4 quota is 1 there). Keep the GKE cluster, disks and model bucket in this region. Zone to be chosen when a spot L4 is actually available. | 2026-11-03 |
| D13 | Python tooling: stdlib `venv` (no conda/uv). Python 3.11 locally (user's venv is 3.11.15; project allows >=3.10). Package `compserve` in `src/`, installed editable. vLLM is NOT a local dependency (no CUDA on Mac); it gets installed on Delta/cloud with a pinned version. | 2026-11-03 |
| D11 | Pace: as fast as possible. Hours per week not specified. | 2026-11-03 |

## 4. Key technical ideas to keep in mind

- Compression only speeds up serving if vLLM has a fast kernel for the format. Unstructured sparsity gives about no speedup.
- Three targets: weight bytes, KV bytes, FLOPs. Each technique attacks one; the gains overlap and do not add up.
- KV per token for Llama 3.1 8B (FP16) = 2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes = about 128 KiB. Llama 3.1 already uses GQA.
- Decode is memory-bandwidth-bound at small batch; prefill is compute-bound.
- Final result is a quality-vs-cost Pareto frontier per workload (chat vs long-context RAG), not a single "winner".
- Isomorphic pruning yields non-uniform per-layer widths, which break vLLM's standard Llama model code (assumes the same size in every layer). Options: restrict to uniform ratios, or register a custom model class. GQA groups must be respected when pruning heads.
- Low-rank keys interact with RoPE (applied after the key projection). Study how DeepSeek MLA handles this (decoupled RoPE) and how vLLM serves it.
- Eviction needs attention scores, which FlashAttention does not materialize, and it conflicts with PagedAttention blocks and prefix caching.
- A100 (Delta) has no native FP8 compute; FP8 variants must be validated on Ada/Hopper (cloud L4).

## 5. Delta budget (48 GPU-hours)

Rule: no Delta use until the pipeline works end-to-end on the Mac with a 1B model.

| Item | Est. hours |
|---|---|
| Guided exploration: vLLM on the 8B, nsys profile, read `/metrics` | ~4 |
| Calibration / compression runs for 5 compressed variants | ~8-12 |
| Quality evals (perplexity + lm-eval tasks) for all 6 variants | ~15-20 |
| Buffer for failed jobs | ~8-10 |

Speed numbers are NOT taken on Delta. Quality numbers are portable across GPUs. Estimates are rough; revise after the first real runs.

## 6. Phases

| Phase | Build | Where | Interview hook |
|---|---|---|---|
| 0 | Repo skeleton; baseline vLLM on one GPU; read vLLM scheduler / block manager | Mac (1B) then Delta | Why is decode memory-bound? How does PagedAttention reduce fragmentation? |
| 1 | Own benchmark harness (TTFT, TPOT/ITL, throughput, goodput at SLO) and quality eval | Mac, then Delta for quality | How do you measure an LLM service? Defining SLOs |
| 2 | Q: reimplement quantization, export to vLLM format, validate against a reference | Mac (1B), Delta (8B) | Quality cost vs. benefit |
| 3 | P: reimplement isomorphic pruning; handle non-uniform widths in vLLM | Mac (1B), Delta (8B) | Structured vs. unstructured; kernel support |
| 4 | L: low-rank KV in vLLM, including RoPE handling | Mac, Delta | KV memory math, batch size vs. throughput |
| 5 | Combined variants (P+Q, Q+L) and the full quality table | Delta | Errors compound, gains overlap |
| 6 | Containerize; K8s on kind, then GPU node pool on chosen cloud with spot | Mac, then cloud | Cold start, readiness probes, model loading, spot preemption |
| 7 | Prometheus + Grafana: vLLM `/metrics`, GPU exporter | cloud | What would you alert on? |
| 8 | Autoscaling on custom metrics (KEDA vs. Prometheus Adapter) | cloud | Why CPU-based HPA is wrong for GPU inference |
| 9 | Load tests (open-loop vs. closed-loop) and cost model ($/1M tokens at p95 SLO) | cloud | Capacity planning |
| 10 | Write-up and demo | | 5-minute project story |
| Stretch | E (token eviction in vLLM), Qwen run, speculative decoding | | |

Order rationale: measure before optimizing; compression results are the differentiator, K8s/observability is generic and comes after.

## 6b. Progress log

- 2026-11-03: GCP quotas done (D6, D12). Repo skeleton, venv, `benchmarks/mock_server.py` (mock OpenAI-style SSE server) done.
- 2026-11-03: `benchmarks/client.py` single-request async client done and verified against the mock: `RequestRecord` dataclass (raw timestamps; TTFT / TPOT / ITL / E2E derived), SSE parsing via `aiter_lines()`, timestamps only on text events. Cross-check: mean(ITL) == TPOT. Not yet done: HTTP status / error handling, `error` field is never set.
- 2026-11-16: `run_open_loop` done and verified (Poisson arrivals, seeded, latency measured from scheduled time to avoid coordinated omission, unlimited httpx connections). Debugged a real bug together: `gather` indented inside the `for` loop made it partly closed-loop (TTFT grew from lateness, not server slowness). Mock server gained `--max-concurrency` (queueing model). Verified: 3 slots, 3 req/s -> TTFT p50 0.21 s; 15 req/s -> p50 1.34 s, max 2.39 s.
- Interview probes still unanswered by me: (a) two tokens in one chunk — effect on TTFT/ITL/TPOT; (b) benchmark reports p99 TTFT 250 ms but users see 3 s stalls — two ways the measurement tool hides it.
- 2026-11-17: percentile summary (TTFT/TPOT/E2E/ITL, p50/p95/p99, printed in ms), rate sweep, and client error handling done and verified: good server -> 0% errors; HTTP 501 server -> 100% errors; closed port -> 100% errors. Sweep result on mock (3 slots, ~7 req/s capacity): p50 TTFT flat until ~6 req/s but p95 rises from 5 req/s (the knee shows in the tail first). Mock caveat: ITL/TPOT stay flat under load because the mock only models admission queueing; real vLLM will degrade (batch growth, preemption).
- 2026-11-18: throughput, goodput (SLO TTFT 0.5 s, TPOT 0.03 s), `save_results` (results/<name>.json + results/raw/), and 11 unit tests (tests/test_metrics.py) done. Decision: drop Delta, use GCP only (D5). Repo to go on private GitHub.
- (older) Next: throughput (requests/s and output tokens/s, offered vs achieved load), goodput at an SLO (proposed TTFT < 0.5 s, TPOT < 0.03 s), longer runs per rate, save results to JSON, then point the harness at real vLLM.

## 7. Open questions

1. ACTION: create the GCP account and request L4 GPU quota now (`GPUS_ALL_REGIONS` >= 1 plus L4 and preemptible/spot L4 quota in one region). New accounts often start at 0 and approval can take days. The $300 trial may not grant GPU quota; a paid (upgraded) account may be needed. Status (2026-11-03): global GPU quota request approved; on-demand L4 quota already 1. Region is us-central1 (D12). Preemptible (spot) L4 quota in us-central1 confirmed = 1 (2026-11-03). Quota setup is DONE. Capacity in a given zone is not guaranteed; pick a zone where a spot L4 actually starts.
2. Quantization method: GPTQ or AWQ?
3. ~~Baseline skills~~ answered 2026-11-03 (0-3): Docker 3, Kubernetes 1, Helm 0, Prometheus/Grafana 0, GPU profiling 0, Slurm/HPC 2. Teaching implication: K8s, Helm, Prometheus/Grafana and profiling need full concept explanations before building; Docker needs none; Slurm needs only Delta specifics.
4. Live demo style: scale-to-zero with cold start, or recorded demo plus on-demand spin-up.
5. Hours per week available.
6. Verify: which GPU architectures vLLM's 2:4 sparse kernels support (not needed unless 2:4 is revisited).
7. Verify current spot prices on the chosen cloud (published figures I found were months old).

## 8. Teaching checkpoints (answer in my own words)

- Q1: At batch size 1, Llama 3.1 8B in FP16 on an L4 (about 300 GB/s bandwidth): what is the lower bound on time per output token? What does 4-bit weight quantization do to it, and why does that advantage shrink at batch 128?
- Q2: Llama 3.1 uses GQA. How much should KV compression help compared with an older multi-head-attention model? In which workload is KV compression still the biggest lever?

My answers go here once given:

- Q1 (simplified after teaching, 2026-11-03): "Batch-size-1 decode is memory-bound; the chef spends the time fetching from the pantry. 4-bit weights give about 4x because there is less to bring." Correct. Refinements to remember: 4x is an upper bound (scales/zero-points make real 4-bit about 4.2-4.5 bits per weight, plus kernel and KV-cache read overhead); floor at FP16 is about 53 ms/token on an L4; advantage shrinks at large batch because weight reads are amortized and compute dominates.
- Q2 (simplified after teaching, 2026-11-03): KV cache full => queue builds; 4-bit weights free memory so more context/users fit. Core answer correct. Correction to remember: a queued new request waits because there is no free KV memory to admit it (vLLM's scheduler admits requests only when blocks are available), not because of a pantry trip for that request. Prefill is compute-bound (calculators ARE busy), unlike decode. The original GQA question (how much KV compression helps with GQA vs older MHA models) is still unanswered; revisit when we start the low-rank KV phase.
