from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import httpx
import asyncio
import json
import random
import numpy as np
from time import perf_counter


@dataclass
class RequestRecord:
    t_send: float
    t_end: float = 0.0
    t_tokens: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        return self.t_tokens[0] - self.t_send if self.t_tokens else None

    @property
    def tpot(self) -> float | None:
        return (self.t_tokens[-1] - self.t_tokens[0]) / (self.completion_tokens - 1) if self.completion_tokens > 1 else None

    @property
    def e2e(self) -> float:
        return self.t_end - self.t_send

    @property
    def itls(self) -> list[float]:
        return [self.t_tokens[i+1] - self.t_tokens[i] for i in range(len(self.t_tokens) - 1)] if len(self.t_tokens) > 1 else []


async def stream_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int, t_scheduled: float | None = None) -> RequestRecord:
    t_tokens = []
    t_send = t_scheduled if t_scheduled is not None else perf_counter()
    completion_tokens = 0
    payload = {"prompt": prompt, "max_tokens": max_tokens, "stream": True, "stream_options":{"include_usage": True}}
    error = None
    try:
        async with client.stream("POST", url, json=payload) as response_stream:
            if response_stream.status_code != 200:
                body = await response_stream.aread()
                error = f"HTTP {response_stream.status_code}: {body[:200].decode(errors='replace')}"
            else:
                async for line in response_stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event and event["choices"] and "text" in event["choices"][0]:
                        t_tokens.append(perf_counter())
                    if "usage" in event:
                        completion_tokens = event["usage"].get("completion_tokens", 0)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        error = f"{type(e).__name__}: {e}"
    if error is None and not t_tokens:
        error = "No tokens received"
    t_end = perf_counter()

    return RequestRecord(t_send=t_send, t_end=t_end, t_tokens=t_tokens, completion_tokens=completion_tokens, error=error)


async def run_open_loop(prompts: list[str], req_rate: float, max_tokens: int, url: str = "http://localhost:8000/v1/completions", seed: int = 0) -> list[RequestRecord]:
    rng = random.Random(seed)
    tasks = []
    limits = httpx.Limits(max_connections=None)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        t_next = perf_counter()
        for prompt in prompts:
            delay = t_next - perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(stream_request(client, url, prompt, max_tokens, t_scheduled=t_next)))
            t_next += rng.expovariate(req_rate)
        records = await asyncio.gather(*tasks)
    return list(records)


def summarise_records(records: list[RequestRecord], slo_ttft: float = 0.5, slo_tpot: float = 0.03) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    ok = [r for r in records if r.error is None]
    error_rate = 1 - len(ok) / len(records) if len(records) > 0 else 0.0
    metric = {
        "ttft": [r.ttft for r in ok if r.ttft is not None], 
        "tpot": [r.tpot for r in ok if r.tpot is not None],
        "e2e": [r.e2e for r in ok],
        "itls": [itls for r in ok for itls in r.itls]
    }

    if len(records) > 1:
        arrival_span = max(r.t_send for r in records) - min(r.t_send for r in records)
        realized_rate = (len(records) - 1) / arrival_span if arrival_span > 0 else 0.0
    else:
        realized_rate = 0.0

    wall = max(r.t_end for r in records) - min(r.t_send for r in records) if records else 0.0
    requests_per_second = len(ok) / wall if wall > 0 else 0.0
    output_tokens_per_second = sum(r.completion_tokens for r in ok) / wall if wall > 0 else 0.0
    results = {}
    throughput = {}
    throughput["requests_per_second"] = requests_per_second
    throughput["output_tokens_per_second"] = output_tokens_per_second
    throughput["error_rate"] = error_rate
    throughput["realized_offered_rate"] = realized_rate

    good = [r for r in ok if r.ttft is not None and r.ttft <= slo_ttft and r.tpot is not None and r.tpot <= slo_tpot]

    goodput = {
        "requests_per_second": len(good) / wall if wall > 0 else 0.0,
        "slo_attainment": len(good) / len(records) if len(records) > 0 else 0.0
    }


    for metric_name, values in metric.items():
        if values:
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            results[metric_name] = {"p50": p50, "p95": p95, "p99": p99}
    return results, throughput, goodput


def save_results(name: str, config: dict, latency: dict, throughput: dict, goodput: dict,
                 records: list[RequestRecord]) -> Path:
    """Write a summary to results/<name>.json and every per-request record to results/raw/<name>.json."""
    out_dir = Path("results")
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {**config, "timestamp": datetime.now().isoformat(timespec="seconds")},
        "latency": latency,
        "throughput": throughput,
        "goodput": goodput,
    }
    summary_path = out_dir / f"{name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "raw" / f"{name}.json").write_text(json.dumps([asdict(r) for r in records]))
    return summary_path


async def main():
    run_name = "mock"  # change per experiment; reusing a name overwrites that run's files
    req_rates = [3.5, 5, 6, 7, 8, 14]
    config = {
        "url": "http://localhost:8000/v1/completions",
        "n_requests": 200,
        "max_tokens": 11,
        "seed": 0,
        "slo_ttft": 0.5,
        "slo_tpot": 0.03,
    }
    prompts = [f"Hello, world {i}" for i in range(config["n_requests"])]
    for req_rate in req_rates:
        print(f"Running benchmark with request rate {req_rate} requests/sec")
        records = await run_open_loop(prompts, req_rate=req_rate, max_tokens=config["max_tokens"],
                                      url=config["url"], seed=config["seed"])
        summary, throughput, goodput = summarise_records(records, config["slo_ttft"], config["slo_tpot"])
        save_results(f"{run_name}_rate{req_rate}", {**config, "nominal_rate": req_rate},
                     summary, throughput, goodput, records)
        for metric_name, values in summary.items():
            print(f"{metric_name:>5}: p50={values['p50'] * 1000:7.1f} ms  p95={values['p95'] * 1000:7.1f} ms  p99={values['p99'] * 1000:7.1f} ms")
        print(f"Error rate: {throughput['error_rate']:.2%}")
        print(f"nominal {req_rate} / realized {throughput['realized_offered_rate']:.2f} -> achieved {throughput['requests_per_second']:.2f} req/s")
        print(f"Output tokens per second: {throughput['output_tokens_per_second']:.2f}")
        print(f"Goodput: {goodput['requests_per_second']:.2f} req/s, SLO attainment: {goodput['slo_attainment']:.2%}")
        

if __name__ == "__main__":
    asyncio.run(main())
