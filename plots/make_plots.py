"""
Generate every figure for slides/report from results/ (run benchmarks.metrics first).

  python -m benchmarks.metrics && python plots/make_plots.py
"""
import json
import os
import pathlib
import re

os.environ["MPLBACKEND"] = "Agg"  # Kaggle sets an inline backend that the venv does not have
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "plots"
LABELS = {"E0_hf_fp16": "HF baseline fp16 (sequential)", "E3_vllm_fp16": "vLLM fp16",
          "E4_vllm_awq": "vLLM AWQ int4", "E6_vllm_fp16_eager": "vLLM fp16, no CUDA graphs",
          "E7a_vllm_fp16_tp2": "vLLM fp16, TP=2 (2 GPUs)", "E7b_vllm_fp16_2replicas": "vLLM fp16, 2 replicas (2 GPUs)"}
COLORS = {"E0_hf_fp16": "#c53030", "E3_vllm_fp16": "#2b6cb0", "E4_vllm_awq": "#2f855a",
          "E6_vllm_fp16_eager": "#805ad5", "E7a_vllm_fp16_tp2": "#dd6b20", "E7b_vllm_fp16_2replicas": "#319795"}
MAIN = ["E0_hf_fp16", "E3_vllm_fp16", "E4_vllm_awq"]
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def line_plot(s, metric, ylabel, fname, configs, logy=False, scale=1.0, hline=None, title=None):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for c in configs:
        g = s[s.config == c].sort_values("users")
        if g.empty:
            continue
        y, e = g[f"{metric}_mean"] * scale, g[f"{metric}_std"].fillna(0) * scale
        ax.errorbar(g.users, y, yerr=e, marker="o", capsize=3, label=LABELS.get(c, c), color=COLORS.get(c))
    ax.set_xscale("log", base=2)
    us = sorted(s[s.config.isin(configs)].users.unique())
    ax.set_xticks(us, [str(u) for u in us])
    ax.minorticks_off() if not logy else None
    if hline:
        ax.axhline(hline[0], ls="--", color="grey")
        ax.text(0.01, hline[0] * 1.08, hline[1], color="grey", fontsize=9,
                transform=ax.get_yaxis_transform())
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("concurrent users")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    print("wrote", fname)


def scaling_plot(s):
    single, tp, rep = "E3_vllm_fp16", "E7a_vllm_fp16_tp2", "E7b_vllm_fp16_2replicas"
    if not all(c in set(s.config) for c in (single, tp, rep)):
        return
    common = sorted(set.intersection(*[set(s[s.config == c].users) for c in (single, tp, rep)]))
    rows = []
    for u in common:
        base = s[(s.config == single) & (s.users == u)].out_tok_per_s_mean.iloc[0]
        for c in (tp, rep):
            v = s[(s.config == c) & (s.users == u)].out_tok_per_s_mean.iloc[0]
            rows.append({"users": u, "config": c, "speedup": v / base, "efficiency": v / base / 2})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "scaling.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.38
    x = np.arange(len(common))
    for i, c in enumerate((tp, rep)):
        d = df[df.config == c]
        axes[0].bar(x + (i - .5) * w, d.speedup, w, label=LABELS[c], color=COLORS[c])
        axes[1].bar(x + (i - .5) * w, d.efficiency, w, label=LABELS[c], color=COLORS[c])
    axes[0].axhline(2, ls="--", color="grey"); axes[0].text(1.0, 2.04, "ideal 2x", color="grey", fontsize=9, ha="right", transform=axes[0].get_yaxis_transform())
    axes[1].axhline(1, ls="--", color="grey")
    for ax, t in zip(axes, ("Speedup vs 1 GPU (throughput)", "Parallel efficiency = speedup / 2")):
        ax.set_xticks(x, [str(u) for u in common]); ax.set_xlabel("concurrent users"); ax.set_title(t)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "scaling_speedup_efficiency.png"); plt.close(fig)
    print("wrote scaling_speedup_efficiency.png")


def static_batching_plot():
    p = RES / "profiling" / "static_batching.json"
    if not p.exists():
        return
    d = [r for r in json.loads(p.read_text()) if not r.get("oom")]
    oom = [r["batch"] for r in json.loads(p.read_text()) if r.get("oom")]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot([r["batch"] for r in d], [r["out_tok_per_s"] for r in d], "o-", color="#2b6cb0", label="tokens/s")
    ax.set_xscale("log", base=2); ax.set_xlabel("static batch size (HF Transformers)"); ax.set_ylabel("output tokens/s")
    ax2 = ax.twinx()
    ax2.plot([r["batch"] for r in d], [r["peak_mem_gib"] for r in d], "s--", color="#c53030", label="peak memory")
    ax2.set_ylabel("peak GPU memory (GiB)")
    t = "Offline static batching: throughput vs memory"
    if oom:
        t += f"  (OOM at batch {oom[0]})"
    ax.set_title(t); ax.grid(alpha=.3)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "static_batching.png"); plt.close(fig)
    print("wrote static_batching.png")


def memory_bar(s):
    if "gpu_mem_peak_mib_mean" not in s:
        return
    cs = [c for c in s.config.unique()]
    peaks = [s[s.config == c].gpu_mem_peak_mib_mean.max() / 1024 for c in cs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([LABELS.get(c, c) for c in cs], peaks, color=[COLORS.get(c, "grey") for c in cs])
    ax.set_xlabel("peak GPU memory used during benchmark (GiB, summed over GPUs)")
    ax.set_title("GPU memory (vLLM pre-allocates KV cache up to gpu_memory_utilization)")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "gpu_memory.png"); plt.close(fig)
    print("wrote gpu_memory.png")


def kv_capacity_plot():
    """KV-cache capacity (tokens) per config, parsed from the vLLM start-up log lines saved in meta.json.
    This (not nvidia-smi memory) is where weight quantisation shows its memory benefit."""
    rows = []
    for meta in sorted((RES / "raw").glob("*/meta.json")):
        m = json.loads(meta.read_text())
        toks = [int(x.replace(",", "")) for l in m.get("kv_cache_log_lines", [])
                for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", l)]
        if toks:
            rows.append((meta.parent.name, sum(toks)))   # summed over servers (replicas)
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh([LABELS.get(c, c) for c, _ in rows], [t / 1000 for _, t in rows],
                   color=[COLORS.get(c, "grey") for c, _ in rows])
    for b, (_, t) in zip(bars, rows):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {t:,}", va="center", fontsize=9)
    ax.set_xlabel("KV-cache capacity (thousand tokens, all GPUs)")
    ax.set_title("How many tokens of context fit in GPU memory")
    ax.set_xlim(0, max(t for _, t in rows) / 1000 * 1.25)
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "kv_capacity.png"); plt.close(fig)
    pd.DataFrame(rows, columns=["config", "kv_tokens"]).to_csv(RES / "kv_capacity.csv", index=False)
    print("wrote kv_capacity.png")


def quant_speedup_plot(s):
    """AWQ / fp16 throughput ratio vs load: large when memory-bound, fades as the batch grows."""
    a, b = s[s.config == "E4_vllm_awq"], s[s.config == "E3_vllm_fp16"]
    if a.empty or b.empty:
        return
    m = a.merge(b, on="users", suffixes=("_awq", "_fp16"))
    m["speedup"] = m.out_tok_per_s_mean_awq / m.out_tok_per_s_mean_fp16
    m["step_ms_awq"], m["step_ms_fp16"] = 1000 * m.tpot_mean_mean_awq, 1000 * m.tpot_mean_mean_fp16
    m[["users", "speedup", "step_ms_fp16", "step_ms_awq"]].to_csv(RES / "quant_speedup.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(m.users, m.speedup, "o-", color=COLORS["E4_vllm_awq"])
    for u, v in zip(m.users, m.speedup):
        ax.annotate(f"{v:.2f}x", (u, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.axhline(1, ls="--", color="grey")
    ax.set_xscale("log", base=2); ax.set_xticks(m.users, [str(u) for u in m.users]); ax.minorticks_off()
    ax.set_xlabel("concurrent users"); ax.set_ylabel("AWQ int4 throughput / fp16 throughput")
    ax.set_title("Quantisation helps most when decode is memory-bound")
    ax.set_ylim(0, max(m.speedup) * 1.15); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "quant_speedup_vs_users.png"); plt.close(fig)
    print("wrote quant_speedup_vs_users.png")


def main():
    s = pd.read_csv(RES / "summary.csv")
    present = [c for c in MAIN if c in set(s.config)]
    line_plot(s, "out_tok_per_s", "output tokens/s", "throughput_vs_users.png", present,
              title="Throughput vs concurrent users")
    line_plot(s, "ttft_p95", "p95 time to first token (s)", "ttft_p95_vs_users.png", present, logy=True,
              hline=(1.0, "SLO 1 s"), title="Tail TTFT vs concurrent users")
    line_plot(s, "tpot_mean", "mean time per output token (ms)", "tpot_vs_users.png", present, scale=1000,
              title="Decode speed per user vs concurrent users")
    line_plot(s, "e2e_p95", "p95 end-to-end latency (s)", "e2e_p95_vs_users.png", present, logy=True,
              title="Tail request latency vs concurrent users")
    if "gpu_util_mean_mean" in s:
        line_plot(s, "gpu_util_mean", "nvidia-smi GPU utilisation (%)", "gpu_util_vs_users.png", present,
                  title="GPU utilisation vs users")
    if "joules_per_token_mean" in s:
        line_plot(s, "joules_per_token", "energy per output token (J)", "energy_vs_users.png", present, logy=True,
                  title="Energy efficiency")
    extra = [c for c in s.config.unique() if c.startswith("E6")]
    if extra:
        line_plot(s, "out_tok_per_s", "output tokens/s", "ablation_cuda_graphs.png", ["E3_vllm_fp16"] + extra,
                  title="Ablation: CUDA graphs")
    scaling_plot(s)
    static_batching_plot()
    kv_capacity_plot()
    quant_speedup_plot(s)
    memory_bar(s)


if __name__ == "__main__":
    main()
