"""
Greedy output agreement between fp16 and a quantised model on our own prompt set.

  python eval/output_agreement.py generate --model Qwen/Qwen2.5-3B-Instruct --tag fp16
  python eval/output_agreement.py generate --model Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq --tag awq
  python eval/output_agreement.py compare --ref fp16 --other awq
"""
import argparse
import difflib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "eval"


def generate(a):
    from vllm import LLM, SamplingParams
    prompts = [json.loads(l)["prompt"] for l in open(ROOT / a.prompts)][:a.n]
    llm = LLM(model=a.model, dtype="half", quantization=a.quantization, max_model_len=2048,
              gpu_memory_utilization=0.85, seed=0)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=a.max_tokens))
    OUT.mkdir(parents=True, exist_ok=True)
    data = [{"prompt": p, "output": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
            for p, o in zip(prompts, outs)]
    (OUT / f"outputs_{a.tag}.json").write_text(json.dumps(data, indent=1))
    print(f"wrote outputs_{a.tag}.json ({len(data)} prompts)")


def compare(a):
    ref = json.loads((OUT / f"outputs_{a.ref}.json").read_text())
    oth = json.loads((OUT / f"outputs_{a.other}.json").read_text())
    exact, first_div, sim = 0, [], []
    for r, o in zip(ref, oth):
        rt, ot = r["token_ids"], o["token_ids"]
        exact += rt == ot
        k = next((i for i, (x, y) in enumerate(zip(rt, ot)) if x != y), min(len(rt), len(ot)))
        first_div.append(k)
        sim.append(difflib.SequenceMatcher(None, r["output"], o["output"]).ratio())
    n = len(ref)
    res = {"ref": a.ref, "other": a.other, "prompts": n, "exact_match_frac": exact / n,
           "mean_tokens_before_first_divergence": sum(first_div) / n,
           "mean_char_similarity": sum(sim) / n}
    print(json.dumps(res, indent=2))
    (OUT / f"agreement_{a.ref}_vs_{a.other}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--quantization", default=None)
    g.add_argument("--tag", required=True)
    g.add_argument("--prompts", default="prompts/dolly_200.jsonl")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--max-tokens", type=int, default=128)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--other", required=True)
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else compare(a)
