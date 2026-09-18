# ACE-Step 1.5 on Pascal GPUs (Tesla P40, GTX 1080 Ti, etc.)

**Generate full songs in ~24 seconds on 2016-era GPUs.** 🎵

This fork adds support for NVIDIA Pascal GPUs (compute capability 6.1) which are
excluded by the default PyTorch builds (cu128/cu130 dropped sm_61 kernels).

## Benchmarks (2× Tesla P40, 24GB each)

| Song Length | Generation Time | Speed |
|-------------|----------------|-------|
| 15 seconds  | **8.3s**       | 1.8× real-time |
| 60 seconds  | **24.2s**      | 2.5× real-time |

Breakdown for a 60s song:
- LM (1.7B, GPU 2): 13.3s (300 tokens @ 44ms/token)
- DiT turbo (8 steps, GPU 1): 3.2s (0.40s/step)
- VAE decode (GPU 1): 5.8s
- Overhead: ~1.9s

## Why Pascal Was Broken (And How We Fixed It)

### The Problem

Modern PyTorch CUDA builds (cu128, cu130) **dropped sm_61 kernel support**.
On Pascal GPUs, `torch.cuda.is_available()` returns `True` but any CUDA kernel
launch fails with `cudaErrorNoKernelImageForDevice`.

Additionally:
- **vLLM** requires compute capability 7.0+ (Volta)
- **transformers 5.x** uses meta-device initialization which breaks `vector_quantize_pytorch`
- The default `requirements.txt` pins `torch==2.10.0+cu128` (no sm_61)

### The Solution

| Component | Default (broken) | Pascal (working) |
|-----------|-----------------|-----------------|
| PyTorch   | 2.10.0+cu128    | **2.5.0+cu121** |
| transformers | 5.x         | **4.57.6**      |
| LM backend | vLLM/nano-vllm | **PyTorch (pt)** |
| GPU split  | single GPU      | **DiT→GPU0, LM→GPU1** |

### Why torch 2.5.0+cu121?

- Built with CUDA 12.1 which **still includes sm_61 PTX kernels**
- cu128+ builds removed sm_61 (NVIDIA dropped Pascal support)
- cu121 requires `libcudart.so.12` (CUDA 12.x runtime) — **not** CUDA 11

### The fp16 Myth on Pascal

Pascal's native fp16 ALU rate is 1/64th of fp32. **This doesn't matter in practice.**
cuBLAS upcasts fp16→fp32 for all GEMM operations. Measured on P40:

```
fp16 matmul: 8.35 TFLOPS
fp32 matmul: 7.84 TFLOPS
```

fp16 is purely a storage format (saves VRAM bandwidth), compute is fp32.

## Requirements

### Hardware
- NVIDIA Pascal GPU (compute capability 6.1): Tesla P40, P40a, P40c, GTX 1070/1080/1080 Ti, Titan X (Pascal)
- 24GB VRAM per GPU recommended (for 1.7B LM + DiT split)
- Minimum: 12GB VRAM (use `offload_to_cpu=True` and 0.6B LM)
- 16GB+ system RAM
- 10GB+ disk for model weights

### Software
- Linux x86_64 (tested on Ubuntu 22.04)
- CUDA 12.x driver (any 12.x — the driver is forward-compatible)
- Python 3.10 or 3.11
- ~10GB disk space for checkpoints

## Quick Setup

### Option A: Automated Script

```bash
./setup_pascal.sh
```

This creates a `pascal` conda env with all correct dependencies.

### Option B: Manual Setup

```bash
# 1. Create environment
conda create -n pascal python=3.10 -y
conda activate pascal

# 2. Install PyTorch with CUDA 12.1 (has sm_61 kernels!)
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install ACE-Step dependencies (Pascal-compatible versions)
pip install -r requirements-pascal.txt

# 4. Download model weights (~9.4GB)
python -c "
from acestep.checkpoint_manager import download_all_checkpoints
download_all_checkpoints('./checkpoints')
"

# 5. Verify
python verify_pascal.py
```

### Downloading Models Manually

If auto-download fails, grab from HuggingFace:

```bash
# Main DiT model (turbo, 8 steps)
huggingface-cli download ACE-Step/Ace-Step1.5 \
    --include "checkpoints/acestep-v15-turbo/*" \
    --local-dir .

# LM (1.7B)
huggingface-cli download ACE-Step/Ace-Step1.5 \
    --include "checkpoints/acestep-5Hz-lm-1.7B/*" \
    --local-dir .

# Text encoder
huggingface-cli download ACE-Step/Ace-Step1.5 \
    --include "checkpoints/Qwen3-Embedding-0.6B/*" \
    --local-dir .

# VAE
huggingface-cli download ACE-Step/Ace-Step1.5 \
    --include "checkpoints/vae/*" \
    --local-dir .
```

## Usage

### Command Line

```bash
conda activate pascal

# Generate a song (single GPU, auto-detect)
python cli.py generate \
    --caption "An upbeat indie rock song with jangly guitars" \
    --duration 60 \
    --steps 8 \
    --seed 42

# Generate with explicit GPU split (recommended for 2× P40)
python cli.py generate \
    --caption "Lo-fi hip hop beat with vinyl crackle" \
    --duration 30 \
    --device cuda:0 \
    --lm-device cuda:1
```

### Python API

```python
import sys
sys.path.insert(0, '.')

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Initialize DiT on GPU 0
dit = AceStepHandler()
dit.initialize_service(
    project_root='.',
    config_path='acestep-v15-turbo',
    device='cuda:0',
    offload_to_cpu=False,
)

# Initialize LM on GPU 1
llm = LLMHandler()
llm.initialize(
    checkpoint_dir='./checkpoints',
    lm_model_path='acestep-5Hz-lm-1.7B',
    backend='pt',          # PyTorch backend (NOT vllm/llamacpp)
    device='cuda:1',
    offload_to_cpu=False,
)

# Generate!
params = GenerationParams(
    task_type='text2music',
    caption='A dreamy ambient electronic piece with pads and arpeggios',
    duration=60,
    inference_steps=8,     # turbo mode
    guidance_scale=1.0,    # no CFG (turbo)
    seed=42,
)
result = generate_music(
    dit, llm,
    params=params,
    config=GenerationConfig(audio_format='wav'),
    save_dir='./output',
)

print(f"Generated: {result.audios[0]['path']}")
```

### HTTP API

The Gradio server also exposes a REST API on the same port (7860):

| Method | Path | Purpose |
|--------|------|----------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/release_task` | **Generate music** (main endpoint) |
| `POST` | `/query_result` | Check task status |
| `GET` | `/v1/audio?task_id=...` | Download generated WAV |
| `POST` | `/format_input` | Enhance lyrics/caption via LM |

#### Generate a song

```bash
curl -X POST http://localhost:7860/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "An upbeat indie rock song with jangly guitars",
    "lyrics": "[Verse 1]\nYour lyrics here...\n[Chorus]\nOh yeah...",
    "duration": 60,
    "bpm": 120,
    "keyscale": "C Major",
    "vocal_language": "en",
    "seed": 42,
    "inference_steps": 8,
    "lm_temperature": 0.85
  }'
```

Response: `{"task_id": "...", "status": "pending"}`

#### Poll for completion

```bash
curl -X POST http://localhost:7860/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_id": "YOUR_TASK_ID"}'
```

#### Download the audio

```bash
curl -o song.wav "http://localhost:7860/v1/audio?task_id=YOUR_TASK_ID"
```

#### Enhance lyrics (optional, before generating)

```bash
curl -X POST http://localhost:7860/format_input \
  -H "Content-Type: application/json" \
  -d '{"caption": "sad piano ballad", "lyrics": "[Verse]\nI miss you..."}'
```

Returns enhanced caption + structured metadata (BPM, key, duration) you can feed into `/release_task`.

## Generation Parameters

### Basic

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | str | — | Music description (genre, mood, instruments, style). < 512 chars |
| `lyrics` | str | `[Instrumental]` | Song lyrics with structure tags like `[Verse 1]`, `[Chorus]`. < 4096 chars |
| `duration` | int | auto | Target length in seconds (10–600) |
| `bpm` | int/None | auto | Beats per minute (30–300) |
| `keyscale` | str | auto | Musical key, e.g. `"C Major"`, `"Am"` |
| `timesignature` | int | auto | Time signature numerator: 2 (2/4), 3 (3/4), 4 (4/4), 6 (6/8) |
| `vocal_language` | str | `"en"` | Language code: en, ja, zh, ko, es, fr, de, etc. |
| `seed` | int | -1 | Reproducibility seed. -1 = random |

### Generation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_steps` | int | 8 (turbo) | Diffusion steps. 8 for turbo, 32–100 for base model |
| `guidance_scale` | float | 1.0 | CFG strength. Only affects non-turbo models |
| `thinking` | bool | False | Enable LM Chain-of-Thought reasoning (better structure, slower) |
| `lm_temperature` | float | 0.85 | LM sampling temperature (0.0–2.0). Higher = more creative |
| `lm_top_k` | int | 0 | LM top-k sampling (0 = disabled) |
| `lm_top_p` | float | 1.0 | LM nucleus sampling (1.0 = disabled) |
| `lm_negative_prompt` | str | `""` | What the LM should avoid generating |
| `use_cot_metas` | bool | True | Let LM generate BPM/key/duration via reasoning |
| `use_cot_caption` | bool | True | Let LM rewrite/expand your caption |

### Advanced (DiT)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_adg` | bool | False | Adaptive Dual Guidance (base model only) |
| `cfg_interval_start` | float | 0.0 | Start ratio for CFG application |
| `cfg_interval_end` | float | 1.0 | End ratio for CFG application |
| `shift` | float | 1.0 | Timestep shift factor |
| `latent_shift` | float | 0.0 | Additive shift on DiT latents before VAE decode |
| `latent_rescale` | float | 1.0 | Multiplicative rescale on latents |
| `enable_normalization` | bool | True | Loudness normalization on output |
| `normalization_db` | float | -1.0 | Target peak loudness in dBFS |

### Task Types

| `task_type` | Description | Extra params needed |
|-------------|-------------|---------------------|
| `text2music` | Standard text-to-music generation | — |
| `cover` | Cover an existing song in new style | `reference_audio` |
| `repaint` | Replace a section of a song | `src_audio`, `repainting_start`, `repainting_end` |
| `extract` | Extract/stem separation | `src_audio` |
| `complete` | Continue/extend a song | `src_audio` |
| `lego` | Splice sections together | `src_audio`, `repainting_start`, `repainting_end` |

### LM CoT (Chain-of-Thought) Modes

| Parameter | Effect |
|-----------|--------|
| `use_cot_metas=True` | LM reasons about BPM, key, duration before generating |
| `use_cot_caption=True` | LM expands your short caption into a detailed description |
| `use_cot_language=True` | LM auto-detects vocal language from lyrics |

## Single GPU Setup (12-16GB VRAM)

If you only have one Pascal GPU:

```python
# Use offloading — slower but works
dit.initialize_service(
    project_root='.',
    config_path='acestep-v15-turbo',
    device='cuda:0',
    offload_to_cpu=True,    # offload LM to CPU between steps
)

llm.initialize(
    checkpoint_dir='./checkpoints',
    lm_model_path='acestep-5Hz-lm-1.7B',
    backend='pt',
    device='cpu',           # run LM on CPU
    offload_to_cpu=False,
)
```

Expected: ~60-90s per 60s song (CPU LM is the bottleneck at ~60ms/token).

## Troubleshooting

### `cudaErrorNoKernelImageForDevice`
Your PyTorch build doesn't have sm_61 kernels. You need `torch 2.5.0+cu121`:
```bash
pip install torch==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### `Tensor on device cpu is not on the expected device meta`
You have transformers 5.x. Downgrade:
```bash
pip install "transformers>=4.51.0,<4.58.0"
```

### `CUDA out of memory`
Split models across GPUs or enable offloading:
```bash
# Check GPU memory
nvidia-smi

# Use the other GPU for LM
python cli.py generate --device cuda:0 --lm-device cuda:1
```

### `NaN or Inf latents` (fp16 overflow)
Pascal GPUs overflow in fp16 during DiT diffusion. Set float32:
```bash
export ACESTEP_DTYPE=float32
```
This is **required** for Pascal. Add it to your `.bashrc` or the launch script.

### `Expected all tensors to be on the same device`
Multi-GPU setup issue — make sure you're using the patched version which
auto-splits DiT→cuda:0, LM→cuda:1 and verifies all weights are on-device.

### `libcudart.so.12 not found`
Install CUDA 12 runtime:
```bash
# Ubuntu
sudo apt install cuda-runtime-12-0
# Or symlink from existing CUDA install
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Slow generation (< 1 song/min)
Make sure you're using the turbo DiT (8 steps), not the full model (50 steps):
```python
config_path='acestep-v15-turbo'  # 8 steps, no CFG
inference_steps=8
guidance_scale=1.0
```

## Architecture Notes

### GPU Memory Layout (2× P40, 24GB each)

**With GPUs otherwise free:**
```
GPU 0 (cuda:0)                    GPU 1 (cuda:1)
┌─────────────────────┐          ┌─────────────────────┐
│ DiT (turbo) ~11.5GB │          │ LM (1.7B) ~3.5GB    │
│ VAE ~0.3GB          │          │ Tokenizer ~1.2GB    │
│ Text encoder ~1.2GB │          │ Constrained proc    │
│ KV cache / actvns   │          │                     │
│ ~10GB free          │          │ ~19GB free          │
└─────────────────────┘          └─────────────────────┘
```

**With other models loaded (e.g. Qwen 27B split across both cards):**
```
GPU 0 (cuda:0)                    GPU 1 (cuda:1)
┌─────────────────────┐          ┌─────────────────────┐
│ Qwen 27B (part)     │          │ Qwen 27B (part)     │
│ DiT (turbo) ~11.5GB │          │ LM (1.7B) ~3.5GB    │
│ VAE + text enc      │          │ Tokenizer ~1.2GB    │
│ ~1GB free (tight!)  │          │ ~8-10GB free        │
└─────────────────────┘          └─────────────────────┘
```

> ⚠️ **Important**: If you have other models loaded on your GPUs (LLM servers,
> ComfyUI, etc.), check free VRAM with `nvidia-smi` before running. The DiT
> needs ~12GB contiguous. If GPU 0 is too full, swap: put DiT on the card
> with more free space, LM on the other. The code supports any `cuda:N` device.

### Why Not vLLM?
vLLM requires compute capability 7.0+ (uses CUDA graphs, flash attention).
The PyTorch backend (`backend='pt'`) uses standard PyTorch inference which
works on any CUDA-capable GPU.

### Why Not llama.cpp?
llama.cpp doesn't have an "acestep-lm" architecture loader. The model uses
custom audio code tokens (`<|audio_code_N|>`) that require the full
transformers tokenizer. The PyTorch backend handles this natively.

## What Was Modified From Upstream

| File | Change |
|------|--------|
| `acestep/gpu_config.py` | Added `_cuda_kernels_work()` — runtime sm_61 kernel verification |
| `acestep/llm_inference.py` | Import kernel check, clean CPU fallback with warning |
| `acestep/core/generation/handler/init_service_loader.py` | `low_cpu_mem_usage=False` for transformers compat |
| `requirements-pascal.txt` | New: Pascal-compatible dependency pins |
| `setup_pascal.sh` | New: one-command setup script |
| `verify_pascal.py` | New: verification script |

The llama.cpp backend code is preserved in `llm_inference.py` for future use
(if someone writes an acestep-lm architecture loader for llama.cpp).

## Contributing

If you get this working on another Pascal variant (GTX 1070, Titan X, etc.),
please open an issue with your benchmarks!

## License

Same as upstream ACE-Step: [Apache 2.0](LICENSE)
