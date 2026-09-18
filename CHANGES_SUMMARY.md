# ACE-Step 1.5 on Pascal GPUs — Changes Summary

## TL;DR

**60-second songs in 24 seconds on 2× Tesla P40 (2016).** Faster than real-time.

The root cause was PyTorch build selection: modern cu128/cu130 builds dropped sm_61
kernels. The fix is using `torch 2.5.0+cu121` which still ships Pascal support.

## What Works

| Component | Config | Performance |
|-----------|--------|-------------|
| DiT (turbo) | cuda:0, fp16, 8 steps | 0.40s/step (60s song) |
| LM (1.7B) | cuda:1, fp16, PyTorch backend | 44ms/token |
| VAE decode | cuda:0, tiled | 5.8s (60s song) |
| **Total** | | **24.2s for 60s song** |

## Root Cause Analysis

1. **PyTorch cu128/cu130 builds**: No sm_61 PTX kernels → `cudaErrorNoKernelImageForDevice`
2. **vLLM**: Requires compute capability 7.0+ → unusable on Pascal
3. **transformers 5.x**: Meta-device init breaks `vector_quantize_pytorch` → need 4.57.x
4. **torch cu118**: Requires `libcudart.so.11` which doesn't exist on modern systems

The working combination: **torch 2.5.0+cu121 + transformers 4.57.x + PyTorch backend**

## Files Added

| File | Purpose |
|------|---------|
| `README-PASCAL.md` | Full setup guide, benchmarks, troubleshooting |
| `requirements-pascal.txt` | Pascal-compatible dependency pins |
| `setup_pascal.sh` | One-command conda env setup |
| `verify_pascal.py` | Verification script (checks kernels, deps, checkpoints) |

## Files Modified

| File | Change | Why |
|------|--------|-----|
| `acestep/gpu_config.py` | Added `_cuda_kernels_work()` | Runtime sm_61 kernel verification (2×2 matmul test, cached) |
| `acestep/llm_inference.py` | Import kernel check, CPU fallback | Clean fallback with warning instead of hang |
| `acestep/core/generation/handler/init_service_loader.py` | `low_cpu_mem_usage=False` | transformers compat (no-op in 4.x, safety for edge cases) |

## Preserved (Not Used)

The llama.cpp backend code in `llm_inference.py` is preserved per design.
It's not viable currently (no "acestep-lm" architecture in llama.cpp) but
would be a fallback if someone implements the loader.

## Key Insight: fp16 on Pascal

Despite Pascal's native fp16 ALU being 1/64th the fp32 rate, **this is irrelevant
for deep learning workloads**. cuBLAS upcasts to fp32 for all GEMM operations.
Measured: fp16 matmul = 8.35 TFLOPS, fp32 matmul = 7.84 TFLOPS (same speed).
fp16 is purely a storage/bandwidth optimization.
