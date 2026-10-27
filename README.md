# LLM Inference — Hands-on Experiments

Practical experiments with GPU-based LLM serving, vLLM, performance benchmarking, and KV caching.

## Projects

| Folder | Experiment | Status |
| --- | --- | --- |
| [01-vllm-setup](01-vllm-setup/) | Serve Qwen2.5-3B-Instruct on RunPod B300 MIG with vLLM | Documented |
| [02-throughput-benchmark](02-throughput-benchmark/) | Streaming concurrency/load benchmark (TTFT, latency, output tokens/sec) | Script included; rerun to generate results |
| [03-kv-cache-experiments](03-kv-cache-experiments/) | VRAM/KV-cache behavior under load | Planned |

## Quick start

1. Start a compatible GPU-backed vLLM server using [01-vllm-setup/serve.sh](01-vllm-setup/serve.sh).
2. Install the client requirements: `python -m pip install -r 02-throughput-benchmark/requirements.txt`
3. Run: `python 02-throughput-benchmark/benchmark.py --concurrency 1 5 10 25 50 --requests-per-level 100 --output 02-throughput-benchmark/results/run.csv`

The benchmark sends streaming requests to `http://127.0.0.1:8000/v1/chat/completions`. It does not launch a model itself.

## Important context

The initial setup used Qwen2.5-3B-Instruct, BF16, vLLM 0.31.0, CUDA 12.8, and a B300 MIG partition. Triton attention and greedy decoding worked around a FlashInfer JIT incompatibility in that environment. The ~26 GiB VRAM observed was vLLM's configured allocation, **not** the model's minimum memory requirement.

This repository provides reproducible code; any previously shared benchmark numbers are observations, not output reproduced by these scripts. GPU availability, model settings, server version, and workload affect results.

## License

No license has been selected yet.
