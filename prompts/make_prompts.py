"""
Build the benchmark prompt sets (commit the resulting .jsonl files: they are your "shared prompts").

  python prompts/make_prompts.py --model Qwen/Qwen2.5-3B-Instruct

Creates:
  prompts/dolly_200.jsonl      realistic: 200 instructions from databricks-dolly-15k (CC BY-SA 3.0),
                               formatted with the model's chat template, seed 42
  prompts/fixed_512.jsonl      controlled: 200 raw WikiText-2 passages of ~512 tokens each
"""
import argparse
import json
import pathlib

from datasets import load_dataset
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-input", type=int, default=1024, help="skip dolly prompts longer than this")
    ap.add_argument("--fixed-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)

    # 1) realistic prompts
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=a.seed)
    out, lens = [], []
    for ex in ds:
        content = ex["instruction"] + (f"\n\nContext:\n{ex['context']}" if ex["context"].strip() else "")
        prompt = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
        n = len(tok(prompt).input_ids)
        if n > a.max_input:
            continue
        out.append({"prompt": prompt, "input_tokens": n, "category": ex["category"]})
        lens.append(n)
        if len(out) == a.n:
            break
    with open(HERE / f"dolly_{a.n}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    lens.sort()
    print(f"dolly_{a.n}.jsonl: input tokens min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}")

    # 2) fixed-length controlled prompts
    text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    ids = tok(text).input_ids
    L = a.fixed_len
    with open(HERE / f"fixed_{L}.jsonl", "w") as f:
        for i in range(a.n):
            chunk = ids[i * L:(i + 1) * L]
            if len(chunk) < L:
                break
            f.write(json.dumps({"prompt": tok.decode(chunk), "input_tokens": L}) + "\n")
    print(f"fixed_{L}.jsonl written")


if __name__ == "__main__":
    main()
