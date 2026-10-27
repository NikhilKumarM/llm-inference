# Experiment 01 — Serve Qwen 3B with vLLM

Deploy `Qwen/Qwen2.5-3B-Instruct` using vLLM on a RunPod NVIDIA B300 MIG GPU.

## Environment used
- RunPod PyTorch CUDA 12.8 container
- NVIDIA B300 SXM6 AC MIG 1g.34gb (31,232 MiB available in `nvidia-smi`)
- Python 3.12, vLLM 0.31.0
- Qwen2.5-3B-Instruct, BF16 (~5.79 GiB model weights)

## Start

```bash
source /root/.virtualenvs/vllm-env/bin/activate
bash 01-vllm-setup/serve.sh
```

If starting fresh, create a virtual environment and install a vLLM version compatible with your CUDA/GPU environment first.

## Test

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Explain Kubernetes in three sentences."}],"temperature":0,"max_tokens":100}'
```

## Lessons

- The API responding to `/v1/models` doesn't prove generation works: test `/v1/chat/completions`.
- FlashInfer attention/sampling JIT compilation failed with the installed CUDA 12.8 toolkit on this B300 MIG setup. Triton attention plus greedy decoding (`temperature=0`) worked.
- `--gpu-memory-utilization 0.85` led to ~26 GiB allocated VRAM. This is **not** the minimum VRAM needed for a 3B model: vLLM also reserves cache/runtime memory.
- `nvidia-smi` showed GPU-Util as N/A for the MIG device even though vLLM was serving requests.
