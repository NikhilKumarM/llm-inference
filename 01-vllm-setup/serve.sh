#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  --enforce-eager
