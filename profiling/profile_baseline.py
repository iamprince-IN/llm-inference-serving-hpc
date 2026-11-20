"""
Profile the HF baseline to identify the bottleneck. Produces the evidence for the
"Performance Analysis -> pre-analysis" section of the report.

  python profiling/profile_baseline.py --model Qwen/Qwen2.5-3B-Instruct

Measures:
 1. Prefill time vs per-token decode time at batch 1 (manual KV-cache loop, exact timings)
 2. MBU (model bandwidth utilisation) and MFU (model FLOPs utilisation) of decode
 3. Offline STATIC batching: tokens/s and peak memory for batch sizes 1..32 (+ OOM point)
 4. PyTorch Profiler operator table + Chrome trace (open in https://ui.perfetto.dev)
Writes results/profiling/*.json|txt
"""
import argparse
import json
import pathlib
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "profiling"


def sync():
    torch.cuda.synchronize()


def make_input(tok, n_tokens, batch=1):
    base = ("High performance computing studies how to make programs use hardware efficiently. "
            "Large language models generate text one token at a time. ") * 200
    ids = tok(base, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.repeat(batch, 1).cuda()


@torch.no_grad()
def prefill_decode_timing(model, tok, prompt_len, gen):
    ids = make_input(tok, prompt_len)
    for _ in range(2):  # warm-up
        model(ids[:, :32], use_cache=True)
    sync(); t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    sync(); prefill = time.perf_counter() - t0
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    steps = []
    for _ in range(gen):
        sync(); t = time.perf_counter()
        out = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        sync(); steps.append(time.perf_counter() - t)
    return prefill, steps


@torch.no_grad()
def static_batching(model, tok, prompt_len, gen, batch_sizes):
    res = []
    for B in batch_sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = make_input(tok, prompt_len, B)
        mask = torch.ones_like(ids)
        try:
            model.generate(input_ids=ids[:, :16], attention_mask=mask[:, :16], max_new_tokens=4, do_sample=False,
                           pad_token_id=tok.eos_token_id)  # warm-up
            sync(); t0 = time.perf_counter()
            model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=gen, min_new_tokens=gen,
                           do_sample=False, pad_token_id=tok.eos_token_id)
            sync(); dt = time.perf_counter() - t0
            r = {"batch": B, "seconds": dt, "out_tok_per_s": B * gen / dt,
                 "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        except torch.cuda.OutOfMemoryError:
            r = {"batch": B, "oom": True}
            torch.cuda.empty_cache()
        print("[static batching]", r)
        res.append(r)
        if r.get("oom"):
            break
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--peak-bw", type=float, default=320e9, help="bytes/s, T4 ~320e9 (use your GPU's datasheet)")
    ap.add_argument("--peak-flops", type=float, default=65e12, help="fp16 tensor FLOP/s, T4 ~65e12")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
    print(f"params={n_params / 1e9:.2f}B weights={w_bytes / 1e9:.2f}GB KV/token={kv_per_token / 1024:.1f}KiB")

    # 1+2: prefill vs decode, MBU / MFU
    prefill, steps = prefill_decode_timing(model, tok, a.prompt_len, a.gen)
    tpot = statistics.median(steps)
    kv_read = kv_per_token * (a.prompt_len + a.gen / 2)  # avg KV bytes read per decode step
    mbu = (w_bytes + kv_read) / tpot / a.peak_bw
    mfu_decode = 2 * n_params / tpot / a.peak_flops
    mfu_prefill = 2 * n_params * a.prompt_len / prefill / a.peak_flops
    timing = {
        "model": a.model, "gpu": torch.cuda.get_device_name(0), "params": n_params, "weight_bytes": w_bytes,
        "kv_bytes_per_token": kv_per_token, "prompt_len": a.prompt_len, "gen": a.gen,
        "prefill_s": prefill, "prefill_tok_per_s": a.prompt_len / prefill,
        "decode_tpot_median_s": tpot, "decode_tok_per_s": 1 / tpot,
        "decode_theoretical_max_tok_per_s": a.peak_bw / (w_bytes + kv_read),
        "MBU_decode": mbu, "MFU_decode": mfu_decode, "MFU_prefill": mfu_prefill,
        "peak_bw": a.peak_bw, "peak_flops": a.peak_flops,
        "time_share_prefill_for_full_request": prefill / (prefill + sum(steps)),
    }
    print(json.dumps(timing, indent=2))
    (OUT / "prefill_decode.json").write_text(json.dumps(timing, indent=2))

    # 3: static batching
    bs = [int(x) for x in a.batch_sizes.split(",")]
    sb = static_batching(model, tok, a.prompt_len, a.gen, bs)
    (OUT / "static_batching.json").write_text(json.dumps(sb, indent=2))

    # 4: PyTorch profiler on one batch-1 request (32 new tokens keeps the trace small)
    ids = make_input(tok, a.prompt_len)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("generate_32_tokens_batch1"):
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                           min_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    (OUT / "torch_profiler_top_ops.txt").write_text(table)
    prof.export_chrome_trace(str(OUT / "baseline_trace.json"))
    print(table)
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
