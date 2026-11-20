"""
WikiText-2 perplexity, computed the SAME way for fp16 and quantised models (via vLLM prompt logprobs),
so the comparison is apples to apples.

  python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct
  python eval/perplexity.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq

Method: test set split into non-overlapping windows of --window tokens; every token except the
first of each window is scored. (Non-overlapping windows give slightly higher PPL than a sliding
window, but the method is identical across precisions, which is what matters.)
Writes results/eval/ppl_<model>.json
"""
import argparse
import json
import math
import pathlib
import time

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=100, help="~100k tokens; enough for a stable estimate")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported late so --help works without vllm

    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text).input_ids
    W = a.window
    windows = [ids[i:i + W] for i in range(0, len(ids) - W + 1, W)][:a.max_windows]

    llm = LLM(model=a.model, dtype="half", quantization=a.quantization, max_model_len=W + 8,
              gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": w} for w in windows], sp)
    nll, n = 0.0, 0
    for w, o in zip(windows, outs):
        for pos, lp in enumerate(o.prompt_logprobs):
            if lp is None:  # first token has no prediction
                continue
            nll -= lp[w[pos]].logprob  # logprob of the actual next token
            n += 1
    ppl = math.exp(nll / n)
    res = {"model": a.model, "quantization": a.quantization, "window": W, "windows": len(windows),
           "tokens_scored": n, "perplexity": ppl, "seconds": time.time() - t0}
    print(json.dumps(res, indent=2))
    out = ROOT / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ppl_{a.model.replace('/', '__')}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
