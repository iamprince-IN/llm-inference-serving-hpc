# RUNBOOK: running everything on Kaggle (free 2x T4)

Total GPU time for the full set of experiments: roughly 5-6 hours. Do it over 3-4 sessions.
The same cells are in `kaggle_runbook.ipynb`; you can import that notebook into Kaggle directly.

## 0. One-time setup
1. Kaggle account -> verify your phone number (required for GPU).
2. Push this folder to a new GitHub repo (public, or private with a token).
3. Kaggle -> Create -> Notebook. In the right panel: **Accelerator = GPU T4 x2**, **Internet = On**.

## 1. Every session starts with these cells

```python
# Cell A: get the code
%cd /kaggle/working
!git clone https://github.com/<YOUR_USERNAME>/llm-inference-serving-hpc.git
%cd /kaggle/working/llm-inference-serving-hpc
!git pull
```
(No GitHub yet? Upload the zip as a Kaggle Dataset, then `!cp -r /kaggle/input/<dataset-name>/llm-inference-serving-hpc /kaggle/working/ && %cd /kaggle/working/llm-inference-serving-hpc`.)

```python
# Cell B: install (5-10 min). If vLLM fails on the T4 later, change this to an older version, e.g. "vllm==0.7.1"
!pip install -q vllm
!pip install -q -r requirements.txt
```

```python
# Cell C: record the hardware (copy the output into NOTES.md: this IS your "hardware specifications" section)
!python scripts/env_check.py
```

## 2. Day 1: feasibility (do this before anything else)

```python
!python scripts/smoke_vllm.py Qwen/Qwen2.5-0.5B-Instruct
!python scripts/smoke_vllm.py Qwen/Qwen2.5-3B-Instruct
!python scripts/smoke_vllm.py Qwen/Qwen2.5-3B-Instruct-AWQ awq
```
All three must print `SMOKE TEST PASSED`. If not: copy the error to Claude. Typical fixes are an older vLLM version, or `--max-model-len 2048`.
When they pass: `!pip freeze | grep -iE "^(vllm|torch|transformers)=="` and write the versions in NOTES.md.

```python
# build the prompt sets (commit the .jsonl files to GitHub afterwards)
!python prompts/make_prompts.py --model Qwen/Qwen2.5-3B-Instruct
```

```python
# quick end-to-end test of the whole benchmark pipeline (~5 min)
!python run_experiment.py configs/E3_vllm_fp16.json --users 1,8 --repeats 1 --duration 30
!python -m benchmarks.metrics --ref-users 8
```

## 3. Main experiments (one cell each; each takes 30-60 min)

```python
!python run_experiment.py configs/E0_hf_fp16.json        # baseline
!python run_experiment.py configs/E3_vllm_fp16.json      # vLLM fp16
!python run_experiment.py configs/E4_vllm_awq.json       # vLLM AWQ int4
```

## 4. Profiling the baseline (~10-15 min)

```python
!python profiling/profile_baseline.py --model Qwen/Qwen2.5-3B-Instruct
```

## 5. Accuracy (~30 min total)

```python
!python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct
!python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq
!python eval/output_agreement.py generate --model Qwen/Qwen2.5-3B-Instruct --tag fp16
!python eval/output_agreement.py generate --model Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq --tag awq
!python eval/output_agreement.py compare --ref fp16 --other awq
```

## 6. Scaling on 2 GPUs (~60 min)

```python
!python run_experiment.py configs/E7a_vllm_fp16_tp2.json
!python run_experiment.py configs/E7b_vllm_fp16_2replicas.json
```

## 7. Optional ablation (~20 min)
```python
!python run_experiment.py configs/E6_vllm_fp16_eager.json
```

## 8. After EVERY session: make plots and download results

```python
!python -m benchmarks.metrics && python plots/make_plots.py
!cat results/headline_table.md
!zip -qr /kaggle/working/results_$(date +%m%d_%H%M).zip results plots
```
Download the zip from the Output panel (right side) and commit `results/` and `plots/` to GitHub.
**Kaggle deletes /kaggle/working when the session ends** unless you save a version.

## Long runs without babysitting
Interactive sessions can stop if the browser is idle. For long experiments, use
**Save Version -> Save & Run All (Commit)**: the notebook runs in the background, and everything in
/kaggle/working becomes the version's output. Keep only the cells you want in that run.

## If something goes wrong
- `results/logs/<config>_server0.log` holds the full server log. Read the last 30 lines: `!tail -n 30 results/logs/E3_vllm_fp16_server0.log`
- Port already in use / GPU memory still full: `!pkill -f "vllm serve"; pkill -f uvicorn; sleep 10; nvidia-smi`
- Paste errors to Claude with the log lines.
