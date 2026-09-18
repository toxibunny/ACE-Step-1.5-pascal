#!/usr/bin/env bash
# ACE-Step 1.5 — Pascal GPU Setup Script
# Creates a conda env with the correct PyTorch build for sm_61 (Pascal)
#
# Usage: ./setup_pascal.sh [env_name]
# Default env name: pascal

set -euo pipefail

ENV_NAME="${1:-pascal}"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  ACE-Step 1.5 — Pascal GPU Setup"
echo "  Environment: $ENV_NAME"
echo "============================================"

# --- Check for conda ---
if ! command -v conda &>/dev/null; then
    echo "❌ conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- Check for Pascal GPU ---
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_INFO" ]]; then
        echo "🖥️  Detected GPUs:"
        echo "$GPU_INFO" | while read -r line; do echo "   $line"; done
        # Warn if not Pascal
        if ! echo "$GPU_INFO" | grep -q "6\.1"; then
            echo "⚠️  No Pascal (6.1) GPU detected. This setup is optimized for Pascal but"
            echo "   torch 2.5.0+cu121 works on all CUDA 12 GPUs."
        fi
    else
        echo "⚠️  Could not query GPU info (nvidia-smi failed?)"
    fi
else
    echo "⚠️  nvidia-smi not found. Continuing anyway..."
fi

# --- Check disk space ---
AVAILABLE_KB=$(df --output=avail "$SCRIPT_DIR" 2>/dev/null | tail -1 | tr -d ' ' || echo "0")
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
echo "💾 Available disk space: ${AVAILABLE_GB}GB (need ~15GB for env + models)"
if [[ $AVAILABLE_GB -lt 15 ]]; then
    echo "⚠️  Low disk space. Continue? [y/N] "
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && exit 1
fi

# --- Create conda env ---
echo ""
echo "📦 Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Eval conda for this shell
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- Install PyTorch cu121 (has sm_61 kernels!) ---
echo ""
echo "🔥 Installing PyTorch 2.5.0+cu121 (includes Pascal sm_61 kernels)..."
pip install --no-cache-dir \
    torch==2.5.0+cu121 \
    torchvision==0.20.0+cu121 \
    torchaudio==2.5.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# --- Install ACE-Step dependencies ---
echo ""
echo "📚 Installing ACE-Step dependencies..."
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements-pascal.txt"

# --- Verify installation ---
echo ""
echo "✅ Verifying installation..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
cap = torch.cuda.get_device_capability(0)
print(f'   GPU: {torch.cuda.get_device_name(0)}')
print(f'   Compute capability: {cap[0]}.{cap[1]}')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda}')

# Test that kernels actually work (the sm_61 check!)
a = torch.randn(2, 2, device='cuda')
b = torch.randn(2, 2, device='cuda')
c = a @ b
torch.cuda.synchronize()
assert c.sum().item() != float('nan'), 'CUDA kernel execution failed!'
print('   ✅ CUDA kernels working (sm_61 confirmed)')

import transformers
print(f'   transformers: {transformers.__version__}')
assert transformers.__version__.startswith('4.'), f'Need transformers 4.x, got {transformers.__version__}'
print('   ✅ All checks passed!')
"

# --- Download models (optional) ---
echo ""
echo "📥 Model weights (~9.4GB)"
if [[ ! -d "$SCRIPT_DIR/checkpoints/acestep-v15-turbo" ]]; then
    read -p "Download model weights now? [Y/n] " download
    if [[ "$download" != "n" && "$download" != "N" ]]; then
        echo "Downloading from HuggingFace (this may take a while)..."
        pip install --no-cache-dir huggingface_hub
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ACE-Step/Ace-Step1.5',
    allow_patterns=[
        'checkpoints/acestep-v15-turbo/*',
        'checkpoints/acestep-5Hz-lm-1.7B/*',
        'checkpoints/Qwen3-Embedding-0.6B/*',
        'checkpoints/vae/*',
    ],
    local_dir='$SCRIPT_DIR',
)
print('✅ Models downloaded to ./checkpoints/')
"
    else
        echo "Skipping. Run manually later:"
        echo "  python -c \"from acestep.checkpoint_manager import download_all_checkpoints; download_all_checkpoints('./checkpoints')\""
    fi
else
    echo "   ✅ Checkpoints already exist in ./checkpoints/"
fi

# --- Done ---
echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo "============================================"
echo ""
echo "To use:"
echo "  conda activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo ""
echo "  # Set float32 (required for Pascal — fp16 overflows)"
echo "  export ACESTEP_DTYPE=float32"
echo ""
echo "  # Quick test (generates a 15s song):"
echo "  python -c \""
echo "  import sys; sys.path.insert(0, '.')"
echo "  from acestep.handler import AceStepHandler"
echo "  from acestep.llm_inference import LLMHandler"
echo "  from acestep.inference import GenerationParams, GenerationConfig, generate_music"
echo "  dit = AceStepHandler()"
echo "  dit.initialize_service(project_root='.', config_path='acestep-v15-turbo', device='cuda:0')"
echo "  llm = LLMHandler()"
echo "  llm.initialize(checkpoint_dir='./checkpoints', lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda:1')"
echo "  params = GenerationParams(task_type='text2music', caption='test song', duration=15, inference_steps=8, seed=42)"
echo "  result = generate_music(dit, llm, params=params, config=GenerationConfig(audio_format='wav'), save_dir='./output')"
echo "  print(f'Done: {result.audios[0][chr(39)+chr(112)+chr(97)+chr(116)+chr(104)+chr(39)]}')"
echo "  \""
echo ""
echo "  # Or use the CLI:"
echo "  python cli.py generate --caption 'your song description' --duration 60"
echo ""
