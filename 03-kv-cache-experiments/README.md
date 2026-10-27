# Experiment 03 — KV cache (planned)

Investigate how context length, concurrency, and vLLM GPU-memory settings affect cache capacity and request latency.

Proposed runs:
1. Hold model, prompt, output length, and hardware constant.
2. Compare `--gpu-memory-utilization` values where supported.
3. Record `nvidia-smi` allocations, vLLM KV-cache metrics, throughput, and TTFT.
4. Repeat trials before attributing performance changes to cache pressure.

No measurements have been collected for this experiment yet.
