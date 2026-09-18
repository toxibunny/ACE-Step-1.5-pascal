#!/usr/bin/env python
"""Verify ACE-Step Pascal GPU setup.

Checks:
1. PyTorch CUDA works (sm_61 kernels present)
2. transformers version is compatible (4.x)
3. All required packages are installed
4. Model checkpoints exist
5. (Optional) Run a quick 5s generation test

Usage:
    python verify_pascal.py            # basic checks
    python verify_pascal.py --generate # also run a 5s generation
"""

import sys
import os
import time

def check(label, condition, detail=""):
    status = "✅" if condition else "❌"
    msg = f" {status} {label}"
    if detail:
        msg += f" ({detail})"
    print(msg)
    return condition

def main():
    generate = "--generate" in sys.argv
    all_ok = True

    print("=" * 50)
    print("  ACE-Step Pascal GPU Verification")
    print("=" * 50)
    print()

    # --- 1. PyTorch + CUDA ---
    print("[1] PyTorch & CUDA")
    try:
        import torch
        all_ok &= check("torch installed", True, torch.__version__)
        all_ok &= check("CUDA available", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            all_ok &= check("GPU detected", True, gpu_name)
            all_ok &= check("Compute capability", True, f"{cap[0]}.{cap[1]}")
            
            if cap == (6, 1):
                print("   ℹ️  Pascal GPU detected — this is the target architecture")
            
            # The critical test: can we actually run a kernel?
            a = torch.randn(64, 64, device='cuda', dtype=torch.float16)
            b = torch.randn(64, 64, device='cuda', dtype=torch.float16)
            c = a @ b
            torch.cuda.synchronize()
            all_ok &= check("CUDA kernel execution (fp16 matmul)", True)
            
            # Also test fp32
            a32 = torch.randn(64, 64, device='cuda', dtype=torch.float32)
            c32 = a32 @ a32
            torch.cuda.synchronize()
            all_ok &= check("CUDA kernel execution (fp32 matmul)", True)
            
            # Multi-GPU check
            ngpu = torch.cuda.device_count()
            if ngpu > 1:
                all_ok &= check("Multi-GPU", True, f"{ngpu} GPUs")
                # Test GPU 1
                t = torch.randn(4, 4, device=f'cuda:1')
                torch.cuda.synchronize()
                all_ok &= check("GPU 1 kernel execution", True)
            else:
                print("   ℹ️  Single GPU — use offload_to_cpu=True for LM")
        else:
            all_ok &= check("CUDA available", False, "torch.cuda.is_available() = False")
    except ImportError:
        all_ok &= check("torch installed", False, "NOT FOUND")
    except Exception as e:
        all_ok &= check("CUDA kernel execution", False, str(e))
    
    print()

    # --- 2. transformers ---
    print("[2] transformers")
    try:
        import transformers
        ver = transformers.__version__
        is_4x = ver.startswith('4.')
        all_ok &= check("transformers installed", True, ver)
        all_ok &= check("Version 4.x (not 5.x)", is_4x, "5.x uses meta device init")
    except ImportError:
        all_ok &= check("transformers installed", False, "NOT FOUND")
    print()

    # --- 3. Required packages ---
    print("[3] Required packages")
    required = [
        ('diffusers', 'diffusers'),
        ('accelerate', 'accelerate'),
        ('einops', 'einops'),
        ('vector_quantize_pytorch', 'vector-quantize-pytorch'),
        ('loguru', 'loguru'),
        ('soundfile', 'soundfile'),
        ('safetensors', 'safetensors'),
        ('scipy', 'scipy'),
    ]
    for module, pip_name in required:
        try:
            mod = __import__(module)
            ver = getattr(mod, '__version__', 'ok')
            all_ok &= check(f"{pip_name}", True, str(ver))
        except ImportError:
            all_ok &= check(f"{pip_name}", False, "NOT FOUND")
    print()

    # --- 4. Checkpoints ---
    print("[4] Model checkpoints")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    expected = [
        ('acestep-v15-turbo/model.safetensors', 'DiT (turbo)'),
        ('acestep-5Hz-lm-1.7B/model.safetensors', 'LM (1.7B)'),
        ('Qwen3-Embedding-0.6B/model.safetensors', 'Text encoder'),
        ('vae/diffusion_pytorch_model.safetensors', 'VAE'),
    ]
    for rel_path, label in expected:
        full = os.path.join(ckpt_dir, rel_path)
        exists = os.path.isfile(full)
        size = os.path.getsize(full) / (1024**3) if exists else 0
        all_ok &= check(f"{label}", exists, f"{size:.1f}GB" if exists else "MISSING")
    print()

    # --- 5. ACE-Step imports ---
    print("[5] ACE-Step code")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from acestep.gpu_config import is_cuda_available
        cuda_ok = is_cuda_available()
        all_ok &= check("gpu_config.is_cuda_available()", cuda_ok)
    except Exception as e:
        all_ok &= check("gpu_config import", False, str(e))
    
    try:
        from acestep.llm_inference import LLMHandler
        all_ok &= check("llm_inference.LLMHandler", True)
    except Exception as e:
        all_ok &= check("llm_inference import", False, str(e))
    
    try:
        from acestep.handler import AceStepHandler
        all_ok &= check("handler.AceStepHandler", True)
    except Exception as e:
        all_ok &= check("handler import", False, str(e))
    print()

    # --- 6. Optional: generation test ---
    if generate and all_ok:
        print("[6] Quick generation test (15s song)")
        try:
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler
            from acestep.inference import GenerationParams, GenerationConfig, generate_music
            
            ngpu = torch.cuda.device_count()
            dit_device = 'cuda:0'
            llm_device = f'cuda:{min(1, ngpu-1)}' if ngpu > 1 else 'cpu'
            
            print(f"   DiT → {dit_device}, LM → {llm_device}")
            
            dit = AceStepHandler()
            msg, ok = dit.initialize_service(
                project_root='.', config_path='acestep-v15-turbo',
                device=dit_device, offload_to_cpu=False,
            )
            assert ok, f"DiT init failed: {msg}"
            print("   ✅ DiT loaded")
            
            llm = LLMHandler()
            msg, ok = llm.initialize(
                checkpoint_dir='./checkpoints',
                lm_model_path='acestep-5Hz-lm-1.7B',
                backend='pt', device=llm_device,
            )
            assert ok, f"LM init failed: {msg}"
            print("   ✅ LM loaded")
            
            params = GenerationParams(
                task_type='text2music', caption='verification test',
                duration=15, inference_steps=8, seed=42,
            )
            t0 = time.time()
            result = generate_music(
                dit, llm, params=params,
                config=GenerationConfig(audio_format='wav'),
                save_dir='./output/verify',
            )
            elapsed = time.time() - t0
            
            if result.success:
                path = result.audios[0].get('path', 'in-memory')
                all_ok &= check("Generation", True, f"{elapsed:.1f}s for 15s song → {path}")
            else:
                all_ok &= check("Generation", False, result.status_message)
        except Exception as e:
            all_ok &= check("Generation", False, str(e))
        print()

    # --- Summary ---
    print("=" * 50)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Ready to generate music!")
    else:
        print("  ❌ SOME CHECKS FAILED — see above")
    print("=" * 50)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
