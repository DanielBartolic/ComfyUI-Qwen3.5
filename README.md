# ComfyUI-Qwen3.5

Custom ComfyUI nodes for the [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) family — unified natively multimodal models with image, video, and text understanding.

Two nodes included:
- **Qwen 3.5** — transformers-based, supports image + video + text, FP16/8-bit/4-bit quantization
- **Qwen 3.5 (GGUF)** — llama.cpp-based, **9x faster** (152 tok/s vs 17 tok/s), uses GGUF quantized models

![ComfyUI-Qwen3.5 Screenshot](screenshot.png)

## Features

- **Image understanding** — describe, analyze, or answer questions about images
- **Video understanding** — summarize or analyze video content (transformers node)
- **Text generation** — pure text tasks (reasoning, writing, coding)
- **Thinking mode** — optional chain-of-thought reasoning before response
- **GGUF inference** — 152 tokens/second via llama.cpp (Q4_K_XL on RTX PRO 6000)
- **Quantization** — FP16, 8-bit, 4-bit (transformers) or GGUF quantizations (Q4-Q8, BF16)
- **CPU support** — both nodes work on CPU (transformers uses FP32, GGUF needs llama.cpp built without CUDA)
- **ComfyUI compatible** — automatically handles `cudaMallocAsync` compatibility

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DanielBartolic/ComfyUI-Qwen3.5.git
```

Then install dependencies for the node(s) you want to use:

### Transformers node only

```bash
pip install -r ComfyUI-Qwen3.5/requirements.txt -r ComfyUI-Qwen3.5/requirements-transformers.txt
```

### GGUF node only

```bash
pip install -r ComfyUI-Qwen3.5/requirements.txt -r ComfyUI-Qwen3.5/requirements-gguf.txt
```

The GGUF node requires [llama.cpp](https://github.com/ggml-org/llama.cpp) — see [Building llama.cpp](#building-llamacpp) below.

### Both nodes

```bash
pip install -r ComfyUI-Qwen3.5/requirements.txt \
            -r ComfyUI-Qwen3.5/requirements-transformers.txt \
            -r ComfyUI-Qwen3.5/requirements-gguf.txt
```

> **Note:** The transformers node automatically sets `HF_DEACTIVATE_ASYNC_LOAD=1` to prevent OOM errors caused by `transformers >= 5.2.0`'s parallel weight loading conflicting with ComfyUI's `cudaMallocAsync` allocator. No special flags needed.

---

## Model Paths (Manual Placement)

Both nodes look for models inside `ComfyUI/models/LLM/`. Models are auto-downloaded on first use, but you can place them manually to avoid downloads.

### Transformers node

Place the full model folder at:

```
ComfyUI/models/LLM/<model-name>/
```

For example, for Qwen3.5-9B:

```
ComfyUI/models/LLM/Qwen3.5-9B/
├── config.json
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── ...
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

The node checks for `config.json` in that directory — if it exists, no download happens.

You can download the model files manually from HuggingFace:
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) → `ComfyUI/models/LLM/Qwen3.5-0.8B/`
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) → `ComfyUI/models/LLM/Qwen3.5-2B/`
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) → `ComfyUI/models/LLM/Qwen3.5-4B/`
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) → `ComfyUI/models/LLM/Qwen3.5-9B/`
- [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) → `ComfyUI/models/LLM/Qwen3.5-27B/`

### GGUF node

Place GGUF files at:

```
ComfyUI/models/LLM/<model-name>-GGUF/
```

For example, for Qwen3.5-9B Q4_K_XL:

```
ComfyUI/models/LLM/Qwen3.5-9B-GGUF/
├── Qwen3.5-9B-UD-Q4_K_XL.gguf      ← the model
└── mmproj-BF16.gguf                  ← vision projector (required for image input)
```

The GGUF filename format is `<model>-<quantization>.gguf`, with `UD-` prefix for Unsloth Dynamic quantizations (XL variants).

Download manually from HuggingFace:
- [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — pick your quantization + `mmproj-BF16.gguf`
- [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)

---

## Building llama.cpp

The GGUF node calls `llama-mtmd-cli` (the multimodal CLI from llama.cpp). You must build it from source.

### With CUDA (GPU — recommended)

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j$(nproc)
cp llama.cpp/build/bin/llama-mtmd-cli /usr/local/bin/
```

### CPU only (no CUDA required)

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --config Release -j$(nproc)
cp llama.cpp/build/bin/llama-mtmd-cli /usr/local/bin/
```

For CPU-only, set `n_gpu_layers` to `0` in the node settings.

> **Note:** Installing the `llama-cpp-python` pip package does **NOT** provide `llama-mtmd-cli`. You must build llama.cpp from source as shown above.

### Verifying the build

After building, confirm the binary works:

```bash
llama-mtmd-cli --version
```

If you get `llama-mtmd-cli not found`, either:
1. Copy the binary to a directory in your PATH: `cp llama.cpp/build/bin/llama-mtmd-cli /usr/local/bin/`
2. Or set the `cli_path` input in the node to the full path: e.g. `/home/user/llama.cpp/build/bin/llama-mtmd-cli`

The node searches for `llama-mtmd-cli` in this order:
1. The `cli_path` input (if set)
2. System PATH
3. `/usr/local/bin/llama-mtmd-cli`
4. `/opt/llama.cpp/build/bin/llama-mtmd-cli`
5. `/workspace/llama.cpp/build/bin/llama-mtmd-cli`

---

## Node: Qwen 3.5

Transformers-based node. Found under **Qwen3.5** in the node menu. Supports image, video, and text.

Works on both GPU (CUDA) and CPU. On CPU, models load in FP32 (slower but functional).

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | dropdown | Qwen3.5-9B | Model size (0.8B / 2B / 4B / 9B / 27B) |
| `prompt` | STRING | required | Text prompt for the model |
| `system_prompt` | STRING | `""` | Optional system prompt |
| `max_tokens` | INT | 4096 | Maximum tokens to generate |
| `temperature` | FLOAT | 1.0 | Sampling temperature |
| `top_p` | FLOAT | 0.95 | Nucleus sampling |
| `top_k` | INT | 20 | Top-K sampling |
| `repetition_penalty` | FLOAT | 1.0 | Repeated token penalty |
| `enable_thinking` | BOOLEAN | True | Enable chain-of-thought reasoning |
| `quantization` | dropdown | FP16 | FP16 / 8-bit / 4-bit |
| `keep_model_loaded` | BOOLEAN | True | Keep model in VRAM between runs |
| `seed` | INT | 1 | Random seed |
| `image` | IMAGE | optional | Single image input |
| `video` | IMAGE | optional | Video frames (batch of images) |
| `frame_count` | INT | 16 | Max frames to sample from video |

### Supported Models

| Model | Parameters | VRAM (FP16) | VRAM (8-bit) | VRAM (4-bit) |
|-------|-----------|-------------|-------------|-------------|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B | ~2 GB | ~1 GB | ~1 GB |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B | ~5 GB | ~3 GB | ~2 GB |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | ~9 GB | ~6 GB | ~4 GB |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.65B | ~20 GB | ~12 GB | ~7 GB |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B | ~56 GB | ~30 GB | ~17 GB |

---

## Node: Qwen 3.5 (GGUF)

llama.cpp-based node. **9x faster** than transformers on GPU. Found under **Qwen3.5** in the node menu. Supports image and text (no video).

Works on both GPU and CPU. For CPU-only, set `n_gpu_layers` to `0` and build llama.cpp without `-DGGML_CUDA=ON`.

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | dropdown | Qwen3.5-9B | Model size (0.8B / 2B / 4B / 9B / 27B) |
| `quantization` | dropdown | Q4_K_XL | GGUF quantization level |
| `prompt` | STRING | required | Text prompt for the model |
| `system_prompt` | STRING | `""` | Optional system prompt |
| `max_tokens` | INT | 4096 | Maximum tokens to generate |
| `temperature` | FLOAT | 0.7 | Sampling temperature |
| `top_p` | FLOAT | 0.8 | Nucleus sampling |
| `top_k` | INT | 20 | Top-K sampling |
| `repeat_penalty` | FLOAT | 1.0 | Repeated token penalty |
| `n_gpu_layers` | INT | 99 | GPU layers to offload (99 = all, **0 = CPU only**) |
| `ctx_size` | INT | 8192 | Context window size |
| `enable_thinking` | BOOLEAN | False | Enable chain-of-thought reasoning |
| `seed` | INT | 1 | Random seed |
| `image` | IMAGE | optional | Image for vision tasks |
| `cli_path` | STRING | `""` | Path to llama-mtmd-cli (auto-detected if empty) |

### GGUF Quantizations

All from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF):

| Quantization | Size | Speed (RTX PRO 6000) |
|-------------|------|---------------------|
| Q4_K_XL (Unsloth Dynamic) | 6.0 GB | ~152 tok/s |
| Q4_K_M | 5.7 GB | ~150 tok/s |
| Q5_K_XL (Unsloth Dynamic) | 6.7 GB | ~130 tok/s |
| Q6_K_XL (Unsloth Dynamic) | 8.8 GB | ~110 tok/s |
| Q8_0 | 9.5 GB | ~90 tok/s |
| BF16 (full precision) | 17.9 GB | ~60 tok/s |

---

## Output (both nodes)

| Output | Type | Description |
|--------|------|-------------|
| `RESPONSE` | STRING | Model's text response (thinking stripped) |
| `THINKING` | STRING | Extracted reasoning content (empty if thinking disabled) |

## Recommended Sampling Parameters

From the [Qwen3.5 README](https://huggingface.co/Qwen/Qwen3.5-9B):

| Mode | Temperature | Top-p | Top-k | Repetition Penalty |
|------|-------------|-------|-------|---------------------|
| **Thinking** | 1.0 | 0.95 | 20 | 1.0 |
| **Instruct** (default) | 0.7 | 0.8 | 20 | 1.0 |

## Troubleshooting

### `llama-mtmd-cli not found`

You need to build llama.cpp from source — see [Building llama.cpp](#building-llamacpp). The `llama-cpp-python` pip package does **not** include this binary.

### `ggml_cuda_init: failed to initialize CUDA` (Blackwell / newer GPUs)

If cmake reports `CMAKE_CUDA_ARCHITECTURES_NATIVE=No CUDA devices found`, specify the architecture manually:

```bash
# Blackwell (RTX PRO 6000, B200, etc.) — SM_120:
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120

# Ada Lovelace (RTX 4090, L40, etc.) — SM_89:
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89

# Hopper (H100, H200) — SM_90:
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
```

This happens in containers where the GPU is available via `nvidia-smi` but not visible to the CUDA runtime during build.

### CPU-only usage

Both nodes work without a GPU:
- **Transformers node**: Just select a model — it auto-detects CPU and uses FP32. 8-bit/4-bit quantization requires a CUDA GPU.
- **GGUF node**: Build llama.cpp without `-DGGML_CUDA=ON` and set `n_gpu_layers` to `0`.

### Models not downloading

If auto-download fails, place models manually — see [Model Paths](#model-paths-manual-placement) above.

## Requirements

**Transformers node:**
- `transformers >= 5.2.0`
- `torch`
- `bitsandbytes` (for quantization, GPU only)
- `accelerate`

**GGUF node:**
- `llama-mtmd-cli` binary (built from [llama.cpp](https://github.com/ggml-org/llama.cpp) source)
- `huggingface-hub` (for auto model downloads)
- `numpy`, `Pillow`, `torch`

## License

Apache-2.0
