# llm-inference-serving-hpc

**High-Performance LLM Inference and Serving (HPC Capstone)**

**Objective.** Deploy an open-source LLM (Qwen2.5-3B-Instruct), identify the performance bottlenecks of a
naive deployment through profiling, and apply HPC optimisation techniques (continuous batching,
PagedAttention KV-cache management, AWQ 4-bit quantisation, multi-GPU scaling) to improve latency,
throughput, memory efficiency and concurrent-user capacity while preserving accuracy.

## Headline results

All runs: Qwen2.5-3B-Instruct, 200 Dolly prompts, 128 output tokens, closed-loop users,
90 s per point, 3 repeats (mean ± std). SLO per request: TTFT ≤ 1 s and TPOT ≤ 100 ms.

**At 32 concurrent users (one T4 unless noted):**

| Configuration | Output tok/s | p95 TTFT (s) | Mean TPOT (ms) | vs baseline |
|---|---|---|---|---|
| E0 HF Transformers fp16, sequential (baseline) | 22.9 ± 0.2 | 173.5 | 42.9 | 1.0× |
| E3 vLLM fp16 (continuous batching + PagedAttention) | 562.2 ± 7.1 | 1.55 | 50.4 | 24.5× |
| E4 vLLM AWQ int4 | 857.4 ± 19.4 | 1.46 | 31.4 | 37.4× |
| E5 vLLM fp16 + prefix caching* | 673.3 ± 2.4 | 0.45 | 45.5 | 29.4× |
| E6 vLLM fp16, no CUDA graphs | 561.0 ± 6.2 | 1.59 | 50.3 | 24.5× |
| E7a vLLM fp16, tensor parallel (2 GPUs) | 893.3 ± 10.8 | 0.88 | 32.6 | 39.0× |
| E7b vLLM fp16, 2 replicas (2 GPUs) | 766.4 ± 12.6 | 0.95 | 37.6 | 33.4× |

**Peak goodput (tokens/s from requests meeting the SLO):**

| HF baseline | vLLM fp16 | vLLM AWQ | 2 replicas | TP=2 | Prefix cache* |
|---|---|---|---|---|---|
| 23 | 376 | 659 | 767 | 1,028 | 1,063 |

**Key findings**
- **Decode is memory-bound:** it uses 0.22% of GPU compute vs 24% in prefill, and 46% of memory bandwidth.
- **Batching:** 32 sequences per step take 50 ms vs 43 ms for one, giving 24.5× throughput.
- **AWQ int4:** about 2.1–2.2× faster at low load, fading to 1.05× at 128 users. KV capacity grows from 176k to 285k tokens (1.62×). WikiText-2 perplexity rises from 9.51 to 10.14 (+6.6%).
- **PagedAttention:** reserving 4,096 tokens per request would fit 43 users; with paging, 128 users used about 30k of 176k KV tokens.
- **Scaling:** within the SLO, TP=2 delivers more goodput than replicas (1,028 vs 767 tok/s). Replicas give the highest raw throughput at 128 users (1,638 vs 1,563 tok/s).
- **Goodput vs throughput:** vLLM fp16 throughput keeps rising to 1,053 tok/s, but goodput peaks at about 16 users.

\*Upper bound: prompts repeat within a run, and the system prompt is shorter than one
64-token KV block. In a same-session control, prefix caching gave 1.03× throughput at
8 users and cut p50 TTFT from 0.26 to 0.15 s.

Full tables: `results/summary.csv`, `results/headline_table.md`. Plots: `plots/`.

## Hardware

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks (free tier), shared VM |
| GPUs | 2× NVIDIA Tesla T4, 15,360 MiB each, PCIe (no NVLink) |
| GPU architecture | Turing, compute capability 7.5: no bf16, FP8 or FlashAttention-2 |
| GPU peak (datasheet) | ≈ 320 GB/s memory bandwidth, ≈ 65 TFLOP/s fp16 tensor, 70 W |
| CPU | 4 vCPUs |
| NVIDIA driver / CUDA | 580.178.04 / CUDA 13.0 (driver); PyTorch built for CUDA 13.2 |
| OS | Linux 6.18, glibc 2.39 |

**Software:** Python 3.12.3 in a `uv` virtual environment (Kaggle's default Python 3.13
was incompatible with vLLM's dependencies), vLLM 0.34.0 (V1 engine, Triton attention
backend), PyTorch 2.14.0, Transformers 5.22.2.

**Observed hardware behaviour:**
- The T4 averaged about 66 W of its 70 W limit under load, and `nvidia-smi` logged about 17 s of software power capping.
- vLLM uses 64-token KV-cache blocks on this GPU.
- Throughput varied by about 3–7% between Kaggle sessions, so comparisons that matter were also re-tested within one session.

## Experiments
| ID | Configuration | Purpose |
|---|---|---|
| E0 | HF Transformers fp16, sequential server | Baseline |
| E1 | HF Transformers fp16, offline static batching (in `profiling/`) | Effect of naive batching |
| E3 | vLLM fp16 (continuous batching + PagedAttention) | Batching + memory management |
| E4 | vLLM AWQ int4 | + quantisation |
| E6 | vLLM fp16 with `--enforce-eager` (optional) | Ablation: CUDA graphs |
| E7a | vLLM fp16, tensor parallel over 2 GPUs | Scaling (model parallel) |
| E7b | vLLM fp16, 2 independent replicas | Scaling (data parallel) |

Workload: 200 prompts from databricks-dolly-15k (seed 42, chat template applied), 128 output tokens
forced with `ignore_eos`, closed-loop users 1-128, 90 s per point, 3 repeats.

## Reproduce
```bash
pip install vllm && pip install -r requirements.txt
python prompts/make_prompts.py
python run_experiment.py configs/E0_hf_fp16.json     # and E3, E4, E7a, E7b
python profiling/profile_baseline.py
python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct
python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq
python -m benchmarks.metrics && python plots/make_plots.py
```
Step-by-step Kaggle instructions: `RUNBOOK.md`. Test the harness without a GPU:
`python run_experiment.py tests/mock_batched.json` (uses a fake server).

## Repository layout
```
baseline/server_hf.py        naive HF server (OpenAI-style streaming API, one request at a time)
benchmarks/loadgen.py        async closed-loop load generator (per-request timestamps)
benchmarks/metrics.py        raw CSV -> TTFT/TPOT/latency percentiles, throughput, goodput, GPU stats
benchmarks/common.py         nvidia-smi monitor, environment capture, server health check
run_experiment.py            start server(s) -> sweep users x repeats -> stop; one command per config
configs/*.json               one file per experiment
prompts/                     prompt-set builder + the exact prompt files used
profiling/profile_baseline.py  prefill vs decode timing, MBU/MFU, static batching, PyTorch profiler
eval/                        perplexity (WikiText-2) and fp16-vs-quantised output agreement
plots/make_plots.py          all figures
results/                     raw and summarised results (never edited by hand)
tests/                       mock server for testing the harness without a GPU
```

## AI assistance
See `AI_USAGE.md`.
