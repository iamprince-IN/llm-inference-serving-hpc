"""
Run one experiment row end-to-end:
  start server(s) -> wait until ready -> for each (users, repeat): run load + log GPU -> stop server(s)

Usage:
  python run_experiment.py configs/E3_vllm_fp16.json
  python run_experiment.py configs/E3_vllm_fp16.json --users 1,4 --repeats 1 --duration 30   # quick smoke test
  python run_experiment.py configs/E3_vllm_fp16.json --no-server   # server already running elsewhere

Outputs (never edit by hand):
  results/raw/<config>/u<users>_r<rep>.csv        one row per request
  results/raw/<config>/u<users>_r<rep>_gpu.csv    nvidia-smi samples
  results/raw/<config>/meta.json                  config + environment + KV-cache lines from server log
  results/logs/<config>_server<i>.log             full server logs
"""
import argparse
import asyncio
import csv
import json
import os
import pathlib
import re
import signal
import subprocess
import time

from benchmarks.common import GpuMonitor, env_info, save_json, wait_for_server
from benchmarks.loadgen import load_prompts, run_load

ROOT = pathlib.Path(__file__).resolve().parent


def start_servers(cfg, logdir):
    procs = []
    for i, s in enumerate(cfg["servers"]):
        env = os.environ.copy()
        env.update({k: str(v) for k, v in s.get("env", {}).items()})
        log = open(logdir / f"{cfg['name']}_server{i}.log", "w")
        print(f"[start] {s['cmd']}  env={s.get('env', {})}")
        p = subprocess.Popen(s["cmd"], shell=True, env=env, cwd=ROOT, stdout=log,
                             stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        procs.append((p, log))
    return procs


def stop_servers(procs):
    for p, _ in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p, log in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        log.close()
    time.sleep(5)  # let the GPU memory be released before the next experiment


def kv_cache_lines(logdir, name):
    """Pull the lines where vLLM reports KV-cache size / max concurrency (wording varies by version)."""
    pat = re.compile(r"(kv cache|gpu blocks|# gpu blocks|maximum concurrency|cache_config|memory profiling)", re.I)
    out = []
    for f in sorted(logdir.glob(f"{name}_server*.log")):
        for line in f.read_text(errors="ignore").splitlines():
            if pat.search(line):
                out.append(f"{f.name}: {line.strip()[:300]}")
    return out


def write_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--users", help="comma list, overrides config")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--no-server", action="store_true", help="don't start/stop servers")
    a = ap.parse_args()

    cfg = json.loads(pathlib.Path(a.config).read_text())
    users_list = [int(x) for x in a.users.split(",")] if a.users else cfg["users"]
    repeats = a.repeats or cfg.get("repeats", 3)
    duration = a.duration or cfg.get("duration", 90)
    name = cfg["name"]
    outdir = ROOT / "results" / "raw" / name
    logdir = ROOT / "results" / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(ROOT / cfg["prompts"])
    urls = [s["url"] for s in cfg["servers"]]
    procs = [] if a.no_server else start_servers(cfg, logdir)
    try:
        for i, u in enumerate(urls):
            t = wait_for_server(u, cfg.get("startup_timeout", 900), procs[i][0] if procs else None)
            print(f"[ready] {u} after {t:.0f}s")
        meta = {"config": cfg, "env": env_info(), "kv_cache_log_lines": kv_cache_lines(logdir, name),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_json(meta, outdir / "meta.json")
        for line in meta["kv_cache_log_lines"]:
            print("[kv]", line)

        for users in users_list:
            for r in range(repeats):
                tag = f"u{users}_r{r}"
                with GpuMonitor(outdir / f"{tag}_gpu.csv"):
                    t0 = time.time()
                    rows = asyncio.run(run_load(urls, cfg["model"], prompts, users, duration,
                                                cfg.get("max_tokens", 128), seed=r))
                write_rows(rows, outdir / f"{tag}.csv")
                ok = [x for x in rows if not x["error"] and x["ttft"] is not None]
                errs = len(rows) - len(ok)
                wall = (max(x["end"] for x in rows) - min(x["start"] for x in rows)) if rows else 0
                tps = sum(x["out_tokens"] for x in ok) / wall if wall else 0
                print(f"[{name}] users={users:>3} rep={r}  requests={len(ok):>4}  errors={errs}  "
                      f"out_tok/s={tps:8.1f}  took={time.time() - t0:.0f}s")
                if errs and errs == len(rows):
                    print("  !! all requests failed, first error:", rows[0]["error"])
    finally:
        if procs:
            stop_servers(procs)
    print(f"[done] raw results in {outdir}")


if __name__ == "__main__":
    main()
