#!/usr/bin/env python3
"""Streaming OpenAI-compatible vLLM concurrency benchmark.

Example:
  python benchmark.py --concurrency 1 5 10 --requests-per-level 100 --output results/run.csv

No GPU is needed on the client; the server endpoint must be reachable.
"""
import argparse
import asyncio
import csv
import json
import math
import pathlib
import statistics
import time

import aiohttp


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    rank = (len(values) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return values[low] + (values[high] - values[low]) * (rank - low)


async def request_once(session, url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token = None
    tokens = None
    error = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
            # SSE may split events across transport chunks; buffer until blank line.
            buffer = ""
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            continue
                        data = json.loads(raw)
                        if data.get("error"):
                            raise RuntimeError(str(data["error"]))
                        usage = data.get("usage")
                        if usage and usage.get("completion_tokens") is not None:
                            tokens = int(usage["completion_tokens"])
                        for choice in data.get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("content") and first_token is None:
                                first_token = time.perf_counter() - start
            if first_token is None:
                raise RuntimeError("No generated text received")
    except Exception as exc:
        error = str(exc)
    return {
        "latency_s": time.perf_counter() - start,
        "ttft_s": first_token,
        "completion_tokens": tokens,
        "error": error,
    }


async def run_level(session, args, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async def worker():
        async with semaphore:
            return await request_once(session, args.url, args.model, args.prompt, args.max_tokens, args.timeout)

    start = time.perf_counter()
    results = await asyncio.gather(*(worker() for _ in range(args.requests_per_level)))
    duration = time.perf_counter() - start
    successes = [r for r in results if not r["error"]]
    measured = [r for r in successes if r["completion_tokens"] is not None]
    latencies = [r["latency_s"] for r in successes]
    ttfts = [r["ttft_s"] for r in successes if r["ttft_s"] is not None]
    token_total = sum(r["completion_tokens"] for r in measured)
    row = {
        "concurrency": concurrency,
        "requests": len(results),
        "successes": len(successes),
        "errors": len(results) - len(successes),
        "duration_s": round(duration, 3),
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else "",
        "p95_latency_s": round(percentile(latencies, .95), 3) if latencies else "",
        "avg_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else "",
        "p95_ttft_s": round(percentile(ttfts, .95), 3) if ttfts else "",
        "output_tokens": token_total if len(measured) == len(successes) else "",
        "output_tokens_per_s": round(token_total / duration, 2) if successes and len(measured) == len(successes) else "",
        "token_usage_coverage": f"{len(measured)}/{len(successes)}",
    }
    print(json.dumps(row))
    if row["errors"]:
        print("  example error:", next(r["error"] for r in results if r["error"]))
    return row


async def main(args):
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("Warming up...")
        warmup = await request_once(session, args.url, args.model, args.prompt, args.max_tokens, args.timeout)
        if warmup["error"]:
            raise SystemExit("Warm-up failed: " + warmup["error"])
        rows = []
        for concurrency in args.concurrency:
            print(f"Testing concurrency={concurrency} ...")
            rows.append(await run_level(session, args, concurrency))
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default="Explain Kubernetes, distributed systems, and caching with examples.")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default="results/run.csv")
    args = parser.parse_args()
    if any(c < 1 for c in args.concurrency) or args.requests_per_level < 1 or args.max_tokens < 1:
        parser.error("Concurrency, requests-per-level, and max-tokens must be positive")
    asyncio.run(main(args))
