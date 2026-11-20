"""
Naive baseline server: HuggingFace Transformers, fp16, ONE request at a time.

It deliberately speaks the same OpenAI-style /v1/completions streaming API as vLLM,
so the exact same load generator measures both systems (fair comparison).

Run:  MODEL=Qwen/Qwen2.5-3B-Instruct uvicorn baseline.server_hf:app --port 8000
"""
import asyncio
import json
import os
import threading
import time

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()

app = FastAPI()
gpu_lock = asyncio.Lock()  # THE defining weakness of the baseline: requests are serialised


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    ignore_eos: bool = True
    stream: bool = True
    model: str | None = None
    temperature: float = 0.0  # baseline always decodes greedily


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


def _gen_kwargs(r: CompletionRequest, inputs, streamer):
    kw = dict(**inputs, max_new_tokens=r.max_tokens, do_sample=False, streamer=streamer,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if r.ignore_eos:
        kw["min_new_tokens"] = r.max_tokens  # force exact output length, like vLLM's ignore_eos
    return kw


@app.post("/v1/completions")
async def completions(r: CompletionRequest):
    if not r.stream:
        async with gpu_lock:
            inputs = tok(r.prompt, return_tensors="pt").to("cuda")
            kw = _gen_kwargs(r, inputs, None)
            kw.pop("streamer")
            out = await asyncio.to_thread(model.generate, **kw)
            new = out[0, inputs.input_ids.shape[1]:]
            return JSONResponse({"choices": [{"text": tok.decode(new, skip_special_tokens=True)}],
                                 "usage": {"prompt_tokens": inputs.input_ids.shape[1],
                                           "completion_tokens": int(new.shape[0])}})

    async def stream():
        async with gpu_lock:  # time spent waiting here is queueing -> counts toward TTFT, as it should
            inputs = tok(r.prompt, return_tensors="pt").to("cuda")
            n_prompt = int(inputs.input_ids.shape[1])
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
            th = threading.Thread(target=model.generate, kwargs=_gen_kwargs(r, inputs, streamer))
            th.start()
            it = iter(streamer)
            while True:
                chunk = await asyncio.to_thread(next, it, None)  # don't block the event loop
                if chunk is None:
                    break
                if chunk:
                    yield f"data: {json.dumps({'choices': [{'text': chunk}]})}\n\n"
            th.join()
        usage = {"prompt_tokens": n_prompt, "completion_tokens": r.max_tokens if r.ignore_eos else None}
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
