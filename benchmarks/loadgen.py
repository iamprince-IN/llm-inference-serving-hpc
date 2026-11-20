"""
Closed-loop load generator: N virtual users, each sends its next request as soon as the
previous one finishes, for `duration` seconds. Works against any OpenAI-compatible
/v1/completions streaming endpoint (vLLM, or our baseline server).

Records per-request timestamps so every metric can be recomputed later.
Multiple base URLs => users are spread round-robin (used for the 2-replica scaling test).
"""
import asyncio
import json
import random
import time

import aiohttp


async def one_request(session, base_url, model, prompt, max_tokens, uid, rows):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
               "temperature": 0, "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    stamps, usage, err = [], None, ""
    try:
        async with session.post(base_url.rstrip("/") + "/v1/completions", json=payload) as resp:
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:  # line by line (SSE)
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("text"):
                        stamps.append(time.perf_counter())
    except Exception as e:  # network errors, server crash, ...
        err = f"{type(e).__name__}: {e}"[:200]
    t1 = time.perf_counter()
    out_tokens = (usage or {}).get("completion_tokens") or max_tokens
    rows.append({
        "user": uid, "url": base_url, "start": t0, "end": t1,
        "ttft": (stamps[0] - t0) if stamps else None,
        "e2e": t1 - t0,
        "chunks": len(stamps),
        "out_tokens": out_tokens,
        "in_tokens": (usage or {}).get("prompt_tokens"),
        "itl_mean": ((stamps[-1] - stamps[0]) / (len(stamps) - 1)) if len(stamps) > 1 else None,
        "itl_max": max((b - a for a, b in zip(stamps, stamps[1:])), default=None),
        "error": err,
    })


async def _user(uid, session, urls, model, prompts, max_tokens, deadline, rows, seed):
    rng = random.Random(seed * 1000 + uid)
    url = urls[uid % len(urls)]
    while time.perf_counter() < deadline:
        await one_request(session, url, model, rng.choice(prompts), max_tokens, uid, rows)


async def run_load(urls, model, prompts, users, duration, max_tokens, seed=0, warmup=2):
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        junk = []  # warm-up requests (CUDA graphs, caches, connections) are discarded
        await asyncio.gather(*[one_request(s, u, model, prompts[0], 8, -1, junk)
                               for u in urls for _ in range(warmup)])
        rows = []
        deadline = time.perf_counter() + duration
        await asyncio.gather(*[_user(u, s, urls, model, prompts, max_tokens, deadline, rows, seed)
                               for u in range(users)])
    return rows


def load_prompts(path):
    with open(path) as f:
        return [json.loads(l)["prompt"] for l in f if l.strip()]
