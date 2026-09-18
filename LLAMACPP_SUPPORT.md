# llama.cpp Backend Support for ACE-Step 1.5

## Overview

This document describes the llama.cpp backend support added to ACE-Step 1.5 for improved compatibility with legacy CUDA GPUs, particularly Pascal architecture GPUs like the Tesla P40.

## Problem Statement

The original ACE-Step 1.5 implementation uses vLLM/nano-vllm for LM inference, which has the following limitations:

1. **Pascal GPU Incompatibility**: vLLM/nano-vllm requires CUDA compute capability 7.0+ (Volta architecture or newer). Pascal GPUs (compute capability 6.1) fall back to CPU mode, resulting in extremely slow generation times (40+ minutes per song).

2. **Windows Triton Issues**: On Windows, vLLM requires Triton, which may not be properly installed or compatible.

3. **Memory Requirements**: vLLM can be memory-intensive, especially for larger models.

## Solution: llama.cpp Backend

llama.cpp is a lightweight, efficient LLM inference engine that:
- Works on older CUDA GPUs (including Pascal)
- Has minimal dependencies
- Supports both GPU and CPU inference
- Provides good performance even on legacy hardware

## Usage

### Command Line

To use the llama.cpp backend, specify `--backend llamacpp` when running ACE-Step:

```bash
# Using CLI
python cli.py --backend llamacpp

# Or with environment variable
ACESTEP_LM_BACKEND=llamacpp python cli.py
```

### Gradio UI

In the Gradio web interface, select "llamacpp" from the backend dropdown menu in the generation settings.

### API Server

For the API server, set the backend via command line or environment variable:

```bash
# Command line
python api_server.py --backend llamacpp

# Environment variable
ACESTEP_LM_BACKEND=llamacpp python api_server.py
```

## Requirements

### Installation

To use the llama.cpp backend, you need to install the `llama-cpp-python` package:

```bash
pip install llama-cpp-python
```

### Model Format

llama.cpp requires models in **GGUF format**. The ACE-Step 5Hz LM models are typically in HuggingFace format (safetensors). You have two options:

#### Option 1: Pre-convert Models (Recommended)

Convert your HuggingFace models to GGUF format before using them:

```bash
# Install the conversion tool
pip install huggingface-hub llama-cpp-python

# Convert a model (example)
python -m llama_cpp.convert \
  --model /path/to/acestep-5Hz-lm-0.6B \
  --outfile /path/to/acestep-5Hz-lm-0.6B.gguf \
  --outtype f32  # or f16, q8_0, etc.
```

Place the resulting `.gguf` file in the model directory alongside the original files.

#### Option 2: Automatic Detection

The system will automatically look for `.gguf` files in the model directory. If found, it will use them. If not found, it will display a warning and suggest conversion.

## Configuration

### GPU Layers

The backend automatically configures the number of GPU layers based on available VRAM:

- **≥24GB VRAM**: All layers on GPU (`n_gpu_layers=100`)
- **16-24GB VRAM**: ~50 layers on GPU
- **8-16GB VRAM**: ~20 layers on GPU
- **<8GB VRAM**: CPU-only mode (`n_gpu_layers=0`)

You can override this by setting the `ACESTEP_LLAMACPP_GPU_LAYERS` environment variable:

```bash
ACESTEP_LLAMACPP_GPU_LAYERS=50 python cli.py --backend llamacpp
```

### Context Length

The maximum context length defaults to 4096 tokens (same as vLLM). This can be adjusted via the `--max-model-len` parameter or by modifying the code.

## Performance Expectations

### Pascal GPUs (Tesla P40, GTX 1080 Ti, etc.)

With llama.cpp on Pascal GPUs:
- **Expected speedup**: 10-20x faster than CPU fallback
- **Estimated generation time**: 2-5 minutes per song (vs 40+ minutes on CPU)
- **Memory usage**: Efficient, works within 24GB VRAM

### Comparison with vLLM

| Backend | Pascal GPU | Modern GPU (Ampere+) | CPU | Memory Usage |
|---------|-----------|---------------------|-----|--------------|
| vLLM | ❌ Falls back to CPU | ✅ Excellent | ❌ Very slow | High |
| llama.cpp | ✅ Good | ✅ Good | ✅ Acceptable | Low-Medium |
| PyTorch | ⚠️ Limited | ✅ Good | ❌ Slow | Medium |

## Automatic Backend Selection

The system now automatically recommends the best backend for your hardware:

1. **Apple Silicon (MPS)**: MLX backend (native acceleration)
2. **Modern NVIDIA (Ampere+, compute capability ≥8.0)**: vLLM backend
3. **Legacy NVIDIA (Pascal, compute capability 6.1)**: llama.cpp backend (if available), otherwise PyTorch
4. **CPU-only**: PyTorch backend

You can always override the automatic selection using the `--backend` parameter.

## Troubleshooting

### "llama.cpp backend is not available"

**Solution**: Install the package:
```bash
pip install llama-cpp-python
```

### "Model needs to be in GGUF format"

**Solution**: Convert your model to GGUF format as described above.

### "CUDA error: invalid device function"

**Solution**: Your PyTorch installation may not support your GPU. Install a compatible version:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Slow performance on GPU

**Solution**: Check that `n_gpu_layers` is set appropriately. Try increasing it:
```bash
ACESTEP_LLAMACPP_GPU_LAYERS=100 python cli.py --backend llamacpp
```

## Implementation Details

### Files Modified

1. **acestep/llm_inference.py**: Added llama.cpp backend support
   - `_is_llamacpp_available()`: Check for llama.cpp installation
   - `_load_llamacpp_model()`: Load model using llama.cpp
   - `_initialize_5hz_lm_llamacpp()`: Initialize backend
   - `_run_llamacpp()`: Generate text using llama.cpp
   - Updated `initialize()`: Added llama.cpp backend selection
   - Updated `generate_from_formatted_prompt()`: Added llama.cpp generation path

2. **acestep/gpu_config.py**: Updated backend recommendations
   - Added `_is_llamacpp_available()` helper
   - Updated `_apply_lm_backend_compatibility_overrides()` to recommend llama.cpp for legacy GPUs

3. **acestep/llm_backend_compat.py**: Added preflight check
   - `get_llamacpp_preflight_warning()`: Check for llama.cpp availability

4. **cli.py**: Added llamacpp to CLI options

5. **profile_inference.py**: Added llamacpp to benchmarking options

6. **API files**: Updated to support llamacpp backend
   - `acestep/api/http/sample_format_routes.py`
   - `acestep/api/http/release_task_models.py`
   - `acestep/api/llm_readiness.py`
   - `acestep/ui/gradio/events/generation/service_init.py`
   - `acestep/ui/gradio/interfaces/generation_defaults.py`

### Backend Priority

The system tries backends in the following order when auto-detecting:

1. User-specified backend (via `--backend` or environment variable)
2. For legacy CUDA GPUs: llama.cpp → vLLM → PyTorch
3. For modern CUDA GPUs: vLLM → llama.cpp → PyTorch
4. For Apple Silicon: MLX → PyTorch

## Benchmarking

To benchmark the llama.cpp backend against other backends:

```bash
# Compare all backends
python profile_inference.py --lm-backend vllm
python profile_inference.py --lm-backend pt
python profile_inference.py --lm-backend llamacpp

# Or use the benchmark mode
python profile_inference.py --mode benchmark
```

## Known Limitations

1. **CFG Support**: The current implementation provides basic CFG support. For full CFG functionality, the constrained decoding processor is used, but native CFG in llama.cpp is limited.

2. **Batch Processing**: Batch processing works but may not be as optimized as vLLM for large batches.

3. **Model Conversion**: Users must manually convert models to GGUF format. Future versions may include automatic conversion.

4. **Quantization**: The backend currently uses the model's native precision. For better performance, consider using quantized GGUF files (e.g., Q8_0, Q4_K_M).

## Future Improvements

1. **Automatic Model Conversion**: Add automatic conversion from HuggingFace to GGUF format on first use.
2. **Better CFG Support**: Implement native CFG support in llama.cpp backend.
3. **Quantization Options**: Add support for different quantization levels via configuration.
4. **Multi-GPU Support**: Add support for multi-GPU inference with llama.cpp.
5. **Memory Optimization**: Fine-tune memory usage for different GPU configurations.

## References

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5)

## Support

For issues or questions related to the llama.cpp backend:

1. Check this documentation first
2. Review the troubleshooting section
3. Open an issue in the ACE-Step repository with details about your GPU and the error message

## License

This implementation is provided under the same MIT license as ACE-Step 1.5.
