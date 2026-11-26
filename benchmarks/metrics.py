"""
Turn raw per-request CSVs into summary metrics.

  python -m benchmarks.metrics                 # -> results/summary_runs.csv  (one row per run)
                                               # -> results/summary.csv       (mean/std over repeats)
                                               # -> results/headline_table.md
"""
import argparse
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent


def summarize_run(csv_path, gpu_csv=None, slo_ttft=1.0, slo_tpot=0.1, gpu_ids=None):
    df = pd.read_csv(csv_path)
    total = len(df)
    df = df[(df["error"].isna()) | (df["error"] == "")]
    df = df[df["ttft"].notna()]
    if df.empty:
        return {"requests": 0, "errors": total}
    wall = df["end"].max() - df["start"].min()
    tpot = (df["e2e"] - df["ttft"]) / (df["out_tokens"] - 1).clip(lower=1)
    good = (df["ttft"] <= slo_ttft) & (tpot <= slo_tpot)
    s = {
        "requests": len(df), "errors": total - len(df), "wall_s": wall,
        "out_tok_per_s": df["out_tokens"].sum() / wall,
        "req_per_s": len(df) / wall,
        "ttft_mean": df["ttft"].mean(), "ttft_p50": df["ttft"].quantile(.5),
        "ttft_p95": df["ttft"].quantile(.95), "ttft_p99": df["ttft"].quantile(.99),
        "tpot_mean": tpot.mean(), "tpot_p95": tpot.quantile(.95),
        "itl_max_p95": df["itl_max"].quantile(.95) if "itl_max" in df else np.nan,
        "e2e_p50": df["e2e"].quantile(.5), "e2e_p95": df["e2e"].quantile(.95), "e2e_p99": df["e2e"].quantile(.99),
        "goodput_req_s": good.sum() / wall, "slo_ok_frac": good.mean(),
        # Little's law check: average number in flight = sum(latency) / wall  (should be ~ users)
        "littles_concurrency": df["e2e"].sum() / wall,
    }
    if gpu_csv and pathlib.Path(gpu_csv).exists():
        g = pd.read_csv(gpu_csv, skipinitialspace=True)
        for c in ("util_gpu", "mem_used_mib", "power_w"):
            g[c] = pd.to_numeric(g[c], errors="coerce")
        # Only count the GPUs the experiment actually used (CUDA_VISIBLE_DEVICES of its servers).
        # Earlier versions averaged in the idle second GPU, which understated power and memory.
        if gpu_ids is not None:
            g = g[g["index"].isin(gpu_ids)]
        if not g.empty:
            # nvidia-smi writes one row per GPU per sample; timestamps differ slightly between GPUs,
            # so align samples by their order within each GPU instead of by timestamp.
            g = g.assign(sample=g.groupby("index").cumcount())
            per_sample = g.groupby("sample").agg(mem=("mem_used_mib", "sum"), power=("power_w", "sum"),
                                                 n=("index", "nunique"))
            per_sample = per_sample[per_sample["n"] == g["index"].nunique()]  # complete samples only
            s.update({"gpu_util_mean": g["util_gpu"].mean(),
                      "gpu_mem_peak_mib": per_sample["mem"].max(),       # summed over GPUs in use
                      "gpu_power_mean_w": per_sample["power"].mean(),    # summed over GPUs in use
                      "n_gpus": int(g["index"].nunique())})
            if s["out_tok_per_s"] > 0:
                s["joules_per_token"] = s["gpu_power_mean_w"] / s["out_tok_per_s"]
    return s


def gpus_used(cfg_dir):
    """GPU indices an experiment used, from CUDA_VISIBLE_DEVICES in its saved config (None = all)."""
    meta = cfg_dir / "meta.json"
    if not meta.exists():
        return None
    import json
    ids = set()
    for srv in json.loads(meta.read_text()).get("config", {}).get("servers", []):
        vis = str(srv.get("env", {}).get("CUDA_VISIBLE_DEVICES", "")).strip()
        if not vis:
            return None
        ids.update(int(x) for x in vis.split(",") if x.strip().isdigit())
    return sorted(ids) or None


def build(raw_dir, slo_ttft, slo_tpot):
    rows = []
    for cfg_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        gpu_ids = gpus_used(cfg_dir)
        for f in sorted(cfg_dir.glob("u*_r*.csv")):
            m = re.fullmatch(r"u(\d+)_r(\d+)\.csv", f.name)
            if not m:
                continue
            s = summarize_run(f, f.with_name(f.stem + "_gpu.csv"), slo_ttft, slo_tpot, gpu_ids)
            s.update({"config": cfg_dir.name, "users": int(m.group(1)), "rep": int(m.group(2))})
            rows.append(s)
    runs = pd.DataFrame(rows)
    num = [c for c in runs.columns if c not in ("config", "users", "rep")]
    agg = runs.groupby(["config", "users"])[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return runs, agg.reset_index()


def headline(summary, ref_users, slo_ttft, baseline="E0_hf_fp16"):
    lines = [f"Reference load: {ref_users} concurrent users (or the closest measured). SLO: p95 TTFT <= {slo_ttft}s\n",
             "| Config | out tok/s | p95 TTFT (s) | mean TPOT (ms) | peak GPU mem (GiB) | max users within SLO | throughput x vs baseline |",
             "|---|---|---|---|---|---|---|"]
    base_tps = None
    rows = []
    for cfg, g in summary.groupby("config"):
        g = g.sort_values("users")
        r = g.iloc[(g["users"] - ref_users).abs().argmin()]
        ok = g[g["ttft_p95_mean"] <= slo_ttft]
        cap = int(ok["users"].max()) if not ok.empty else 0
        mem = r.get("gpu_mem_peak_mib_mean", np.nan) / 1024
        rows.append((cfg, r, cap, mem))
        if cfg == baseline:
            base_tps = r["out_tok_per_s_mean"]
    for cfg, r, cap, mem in rows:
        x = f"{r['out_tok_per_s_mean'] / base_tps:.1f}x" if base_tps else "n/a"
        lines.append(f"| {cfg} (u={int(r['users'])}) | {r['out_tok_per_s_mean']:.1f} ± {np.nan_to_num(r['out_tok_per_s_std']):.1f} | "
                     f"{r['ttft_p95_mean']:.2f} | {1000 * r['tpot_mean_mean']:.1f} | {mem:.1f} | {cap} | {x} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--slo-ttft", type=float, default=1.0)
    ap.add_argument("--slo-tpot", type=float, default=0.1)
    ap.add_argument("--ref-users", type=int, default=32)
    a = ap.parse_args()
    runs, summary = build(pathlib.Path(a.raw), a.slo_ttft, a.slo_tpot)
    runs.to_csv(ROOT / "results" / "summary_runs.csv", index=False)
    summary.to_csv(ROOT / "results" / "summary.csv", index=False)
    md = headline(summary, a.ref_users, a.slo_ttft)
    (ROOT / "results" / "headline_table.md").write_text(md + "\n")
    print(md)
    print("\nwrote results/summary_runs.csv, results/summary.csv, results/headline_table.md")
