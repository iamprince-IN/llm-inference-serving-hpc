"""
CPU-only fake LLM server for testing the benchmark harness on a laptop (no GPU, no model).

  MOCK_MODE=sequential uvicorn tests.mock_server:app --port 8000   # behaves like the baseline
  MOCK_MODE=batched    uvicorn tests.mock_server:app --port 8000   # behaves like continuous batching

Step time grows slightly with the number of active requests, roughly like a real batched engine.
"""
import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODE = os.environ.get("MOCK_MODE", "batched")
BASE_STEP = float(os.environ.get("MOCK_STEP", "0.01"))
app = FastAPI()
lock = asyncio.Lock()
active = 0


class Req(BaseModel):
    prompt: str
    max_tokens: int = 128
    model: str | None = None


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "mock"}]}


@app.post("/v1/completions")
async def completions(r: Req):
    async def gen():
        global active
        if MODE == "sequential":
            await lock.acquire()
        active += 1
        try:
            await asyncio.sleep(BASE_STEP * 3)  # "prefill"
            for i in range(r.max_tokens):
                await asyncio.sleep(BASE_STEP * (1 + 0.02 * active))
                yield f"data: {json.dumps({'choices': [{'text': f' t{i}'}]})}\n\n"
        finally:
            active -= 1
            if MODE == "sequential":
                lock.release()
        yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': r.max_tokens}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
