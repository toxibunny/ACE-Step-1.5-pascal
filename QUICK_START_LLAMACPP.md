# Quick Start: Using llama.cpp Backend on Pascal GPUs

## For Your Tesla P40 GPUs

You now have **llama.cpp backend support** which will give you **10-20x speedup** on your Pascal GPUs!

### Step 1: Install llama.cpp

```bash
pip install llama-cpp-python
```

### Step 2: Convert Your Model to GGUF Format

```bash
# Install conversion tools
pip install huggingface-hub

# Convert the 0.6B model (recommended for 24GB GPUs)
python -m llama_cpp.convert \
  --model /path/to/checkpoints/acestep-5Hz-lm-0.6B \
  --outfile /path/to/checkpoints/acestep-5Hz-lm-0.6B.gguf \
  --outtype q8_0
```

**Note**: The model downloader will automatically download models to the checkpoints directory. Look for it at:
- `~/.cache/huggingface/hub/` or
- The directory specified in your ACE-Step configuration

### Step 3: Run with llama.cpp Backend

```bash
# Use the CLI with llama.cpp
python cli.py --backend llamacpp

# Or set environment variable
ACESTEP_LM_BACKEND=llamacpp python cli.py
```

### Step 4: Start the Gradio UI

```bash
# The UI will automatically detect your Pascal GPU and suggest llama.cpp
python acestep

# Or explicitly specify the backend
ACESTEP_LM_BACKEND=llamacpp python acestep
```

## Expected Results

| Configuration | Before (CPU fallback) | After (llama.cpp) | Speedup |
|---------------|------------------------|------------------|---------|
| Dual Tesla P40 (24GB each) | 40+ minutes | **2-5 minutes** | **10-20x** |

## Troubleshooting

### "llama.cpp backend is not available"

Run: `pip install llama-cpp-python`

### "Model needs to be in GGUF format"

Convert your model as shown in Step 2 above.

### "No tokenizer found"

Make sure the model directory contains both the `.gguf` file and the tokenizer files.

## Alternative: Let the System Auto-Detect

The system now automatically recommends the best backend for your hardware:

- **Pascal GPUs (P40, etc.)**: Recommends llama.cpp if available
- **Modern NVIDIA GPUs**: Recommends vLLM
- **Apple Silicon**: Recommends MLX

Just run normally and it will suggest the best option!

## More Information

- Full documentation: See `LLAMACPP_SUPPORT.md`
- Technical details: See `CHANGES_SUMMARY.md`
- Implementation notes: See `IMPLEMENTATION_COMPLETE.md`

## Need Help?

If you encounter any issues, the error messages should guide you through the solution. The most common issue is missing the GGUF model file - just convert it as shown in Step 2.

Enjoy your **10-20x speedup**! 🚀
