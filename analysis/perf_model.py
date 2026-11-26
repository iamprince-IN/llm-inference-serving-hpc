"""
Two-term performance model of a vLLM decode step, fitted to the measured results.

    step time (= time per output token, TPOT)  ≈  t0  +  k × B

    t0 : fixed cost per step, shared by the whole batch (dominated by reading all weights once)
    k  : extra cost per concurrent sequence (attention over its KV cache, sampling, scheduling)
    B  : concurrent users (closed loop, so ≈ decode batch size)

Derived quantities
    effective bandwidth   = weight bytes / t0          (how fast weights really stream)
    critical batch  B*    = t0 / k                     (beyond it, per-sequence work dominates)
    throughput(B)         = B / (t0 + k B)  ->  1 / k  (asymptotic ceiling)
    speedup A vs B        = (t0_B + k_B B) / (t0_A + k_A B)

    python -m benchmarks.metrics && python analysis/perf_model.py
Writes results/perf_model.csv, results/perf_model.md, plots/perf_model.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Weight bytes per configuration, from vLLM's "Model loading took X GiB" log line.
GIB = 2 ** 30
WEIGHT_BYTES = {"E3_vllm_fp16": 5.79 * GIB, "E4_vllm_awq": 1.95 * GIB,
                "E7a_vllm_fp16_tp2": 5.79 * GIB / 2}          # per GPU under TP=2
PEAK_BW = 320e9          # T4 datasheet, bytes/s
PEAK_FLOPS = 65e12       # T4 fp16 tensor, FLOP/s
PARAMS = 3.09e9          # Qwen2.5-3B
LAYERS = 36
# E3's 1-user run was a session outlier; replace it by the same-session control (C3) if available.
OUTLIER_FIX = {("E3_vllm_fp16", 1): "C3_control_nocache"}


def tpot_table(summary):
    rows = {}
    for cfg in ("E3_vllm_fp16", "E4_vllm_awq", "E7a_vllm_fp16_tp2"):
        d = summary[summary.config == cfg].set_index("users")["tpot_mean_mean"] * 1000
        for (c, u), src in OUTLIER_FIX.items():
            if c == cfg and u in d.index and ((summary.config == src) & (summary.users == u)).any():
                d.loc[u] = summary[(summary.config == src) & (summary.users == u)]["tpot_mean_mean"].iloc[0] * 1000
        rows[cfg] = d.sort_index()
    return rows


def fit(series):
    u, t = series.index.values.astype(float), series.values
    k, t0 = np.polyfit(u, t, 1)
    pred = t0 + k * u
    r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return t0, k, r2


def main():
    summary = pd.read_csv(RES / "summary.csv")
    tp = tpot_table(summary)
    fits, out = {}, []
    for cfg, s in tp.items():
        t0, k, r2 = fit(s)
        fits[cfg] = (t0, k)
        bw = WEIGHT_BYTES[cfg] / (t0 / 1000)
        out.append({"config": cfg, "t0_ms": t0, "k_ms_per_seq": k, "r2": r2,
                    "effective_bw_GBs": bw / 1e9, "bw_frac_of_peak": bw / PEAK_BW,
                    "critical_batch": t0 / k, "throughput_ceiling_tok_s": 1000 / k})
    df = pd.DataFrame(out)
    df.to_csv(RES / "perf_model.csv", index=False)

    f, a = fits["E3_vllm_fp16"], fits["E4_vllm_awq"]
    lines = ["# Two-term decode model: step time = t0 + k × users\n",
             "| Config | t0 (ms) | k (ms/user) | R² | effective BW (GB/s) | % of peak BW | critical batch t0/k |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['config']} | {r['t0_ms']:.1f} | {r['k_ms_per_seq']:.3f} | {r['r2']:.3f} | "
                     f"{r['effective_bw_GBs']:.0f} | {100 * r['bw_frac_of_peak']:.0f}% | {r['critical_batch']:.0f} |")

    # 1. AWQ speedup predicted by the model vs measured
    lines += ["\n## AWQ vs fp16 speedup: model vs measured\n", "| Users | Model | Measured |", "|---|---|---|"]
    meas = summary.pivot_table(index="users", columns="config", values="out_tok_per_s_mean")
    for B in (8, 16, 32, 64, 128):
        pred = (f[0] + f[1] * B) / (a[0] + a[1] * B)
        m = meas.loc[B, "E4_vllm_awq"] / meas.loc[B, "E3_vllm_fp16"]
        lines.append(f"| {B} | {pred:.2f}× | {m:.2f}× |")
    cross = (f[0] - a[0]) / (a[1] - f[1]) if a[1] > f[1] else float("inf")
    lines.append(f"\nPredicted crossover (fp16 faster than AWQ) above ≈ {cross:.0f} concurrent sequences "
                 "(prediction, beyond the measured range).")

    # 2. Tensor-parallel communication overhead
    t = fits["E7a_vllm_fp16_tp2"]
    comm = t[0] - f[0] / 2
    lines.append(f"\n## Tensor parallel overhead\nIdeal TP=2 fixed cost {f[0] / 2:.1f} ms; measured {t[0]:.1f} ms "
                 f"→ {comm:.1f} ms per step ≈ {1000 * comm / (2 * LAYERS):.0f} µs per all-reduce "
                 f"({2 * LAYERS} all-reduces per step). Per-user cost {t[1]:.3f} vs {f[1]:.3f} ms "
                 f"(ideal {f[1] / 2:.3f}).")

    # 3. Binding SLO constraint
    slo_tpot = 100.0
    lines.append(f"\n## Which SLO binds?\nTPOT alone would allow ≈ {(slo_tpot - f[0]) / f[1]:.0f} users (fp16) and "
                 f"≈ {(slo_tpot - a[0]) / a[1]:.0f} users (AWQ) before TPOT > {slo_tpot:.0f} ms; measured goodput peaks far "
                 "earlier, so first-token latency (TTFT) is the binding constraint.")

    # 4. Utilisation at the largest batch
    B = 128
    step = (f[0] + f[1] * B) / 1000
    mfu = B * 2 * PARAMS / step / PEAK_FLOPS
    kv_read = B * 238 * 36864 / PEAK_BW * 1000
    lines.append(f"\n## At {B} users (fp16)\nStep {1000 * step:.0f} ms; compute utilisation (MFU) ≈ {100 * mfu:.0f}%; "
                 f"reading all KV caches at peak bandwidth would take only ≈ {kv_read:.1f} ms. "
                 "The per-user cost is therefore not KV bandwidth but per-sequence kernel and software overhead.")

    # 5. Energy per token (needs the corrected metrics.py)
    if "joules_per_token_mean" in summary:
        e = summary.pivot_table(index="users", columns="config", values="joules_per_token_mean")
        lines += ["\n## Energy per output token (J)\n", "| Users | " + " | ".join(e.columns) + " |",
                  "|---|" + "---|" * len(e.columns)]
        for B in e.index:
            lines.append(f"| {B} | " + " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in e.loc[B]) + " |")

    (RES / "perf_model.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # plot: measured vs model
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"E3_vllm_fp16": "#12A08A", "E4_vllm_awq": "#E8A33D", "E7a_vllm_fp16_tp2": "#5468D4"}
    names = {"E3_vllm_fp16": "fp16", "E4_vllm_awq": "AWQ int4", "E7a_vllm_fp16_tp2": "fp16 TP=2"}
    for cfg, s in tp.items():
        t0, k = fits[cfg]
        uu = np.linspace(1, s.index.max(), 200)
        ax.plot(s.index, s.values, "o", color=colors[cfg], label=f"{names[cfg]} measured")
        ax.plot(uu, t0 + k * uu, "-", color=colors[cfg], alpha=.6,
                label=f"{names[cfg]} model: {t0:.1f} + {k:.2f}·B ms")
    ax.set_xlabel("concurrent users (B)"); ax.set_ylabel("time per output token (ms)")
    ax.set_title("Decode step time = fixed weight-read cost + per-user cost")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / "perf_model.png", dpi=150)
    print("\nwrote results/perf_model.csv, results/perf_model.md, plots/perf_model.png")


if __name__ == "__main__":
    main()
