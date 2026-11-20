# llm-inference-serving-hpc

**High-Performance LLM Inference and Serving (HPC Capstone)**

**Objective.** Deploy an open-source LLM (Qwen2.5-3B-Instruct), identify the performance bottlenecks of a
naive deployment through profiling, and apply HPC optimisation techniques (continuous batching,
PagedAttention KV-cache management, AWQ 4-bit quantisation, multi-GPU scaling) to improve latency,
throughput, memory efficiency and concurrent-user capacity while preserving accuracy.

## Headline results
_To be filled from `results/headline_table.md` after the experiments._

## Hardware
_To be filled from `scripts/env_check.py` output (GPU, memory, CPU, driver, CUDA, library versions)._

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
