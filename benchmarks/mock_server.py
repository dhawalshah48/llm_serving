"""Mock OpenAI-compatible streaming server for testing the benchmark client without a GPU.

Mimics the part of vLLM's /v1/completions API that the client needs: Server-Sent Events,
one chunk per token, a final usage chunk, then `data: [DONE]`.

Run:
    python benchmarks/mock_server.py --ttft-ms 200 --tpot-ms 20
"""

import argparse
import asyncio
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
CONFIG = {"ttft_ms": 200.0, "tpot_ms": 20.0}
SLOTS = asyncio.Semaphore(10**9)  # replaced at startup by --max-concurrency


def sse(payload: dict | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}\n\n"


@app.post("/v1/completions")
async def completions(request: dict):
    max_tokens = int(request.get("max_tokens", 16))
    include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
    prompt_tokens = len(str(request.get("prompt", "")).split())

    async def stream():
        # Waiting here is the queue: requests past max_concurrency wait for a free slot,
        # which shows up as extra TTFT, like requests waiting for KV-cache space in vLLM.
        async with SLOTS:
            async for chunk in generate():
                yield chunk

    async def generate():
        await asyncio.sleep(CONFIG["ttft_ms"] / 1000)  # stands in for prefill
        for i in range(max_tokens):
            if i > 0:
                await asyncio.sleep(CONFIG["tpot_ms"] / 1000)  # stands in for one decode step
            last = i == max_tokens - 1
            yield sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": f" tok{i}",
                            "finish_reason": "length" if last else None,
                        }
                    ]
                }
            )
        if include_usage:
            yield sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_tokens + max_tokens,
                    },
                }
            )
        yield sse("[DONE]")

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttft-ms", type=float, default=200.0)
    parser.add_argument("--tpot-ms", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=10**9)
    args = parser.parse_args()
    CONFIG["ttft_ms"] = args.ttft_ms
    CONFIG["tpot_ms"] = args.tpot_ms
    SLOTS = asyncio.Semaphore(args.max_concurrency)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
