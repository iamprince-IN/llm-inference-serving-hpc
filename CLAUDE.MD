# Project context
HPC capstone (solo): "High-Performance LLM Inference and Serving".
Presentation: 6 Dec 2026 (7 min + 3 min viva). Results freeze: 26 Nov 2026.
Grading rewards depth of analysis and clear explanation, not raw numbers.

## Setup
- Model: Qwen/Qwen2.5-3B-Instruct (fp16) and Qwen/Qwen2.5-3B-Instruct-AWQ (int4)
- Experiments run on Kaggle 2x T4 (Turing, sm75: no bf16, no FP8, no FlashAttention-2).
  The local machine has NO GPU; test the harness locally with tests/mock_*.json only.
- Experiments: E0 HF baseline, E3 vLLM fp16, E4 vLLM AWQ, E7a TP=2, E7b 2 replicas (configs/).
- See README.md, RUNBOOK.md, PLAN.md.

## Rules
- Keep main runnable; risky changes go on a branch.
- Never edit files in results/raw by hand; never commit tokens or model weights.
- Explain every change in plain language: I must be able to defend all code in the viva.
- Log notable changes and failures in NOTES.md.