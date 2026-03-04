# ComfyUI-Qwen3.5

Custom ComfyUI node for the [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35-68417de654900a475d940ded) family — unified natively multimodal models with image, video, and text understanding.

## Features

- **Image understanding** — describe, analyze, or answer questions about images
- **Video understanding** — summarize or analyze video content
- **Text generation** — pure text tasks (reasoning, writing, coding)
- **Thinking mode** — optional chain-of-thought reasoning before response
- **Quantization** — FP16, 8-bit, or 4-bit to fit different VRAM budgets

## Supported Models

| Model | Parameters | VRAM (FP16) | VRAM (8-bit) | VRAM (4-bit) |
|-------|-----------|-------------|-------------|-------------|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B | ~2 GB | ~1 GB | ~1 GB |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B | ~5 GB | ~3 GB | ~2 GB |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | ~9 GB | ~6 GB | ~4 GB |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.65B | ~20 GB | ~12 GB | ~7 GB |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B | ~56 GB | ~30 GB | ~17 GB |

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DanielBartolic/ComfyUI-Qwen3.5.git
pip install -r ComfyUI-Qwen3.5/requirements.txt
```

The model will be automatically downloaded to `ComfyUI/models/LLM/Qwen3.5-9B/` on first use.

**Important:** Start ComfyUI with `--disable-cuda-malloc` to avoid OOM errors during model loading:

```bash
python main.py --listen 0.0.0.0 --disable-cuda-malloc
```

This is required because `transformers >= 5.2.0` loads model weights in parallel, which conflicts with ComfyUI's default `cudaMallocAsync` memory allocator.

## Node: Qwen 3.5

Found under **🧪AILab/Qwen3.5** in the node menu.

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | dropdown | Qwen3.5-9B | Model size (0.8B / 2B / 4B / 9B / 27B) |
| `prompt` | STRING | required | Text prompt for the model |
| `system_prompt` | STRING | `""` | Optional system prompt |
| `max_tokens` | INT | 4096 | Maximum tokens to generate |
| `temperature` | FLOAT | 0.7 | Sampling temperature |
| `top_p` | FLOAT | 0.8 | Nucleus sampling |
| `top_k` | INT | 20 | Top-K sampling |
| `repetition_penalty` | FLOAT | 1.0 | Repeated token penalty |
| `enable_thinking` | BOOLEAN | False | Enable chain-of-thought reasoning |
| `quantization` | dropdown | FP16 | FP16 / 8-bit / 4-bit |
| `keep_model_loaded` | BOOLEAN | True | Keep model in VRAM between runs |
| `seed` | INT | 1 | Random seed |
| `image` | IMAGE | optional | Single image input |
| `video` | IMAGE | optional | Video frames (batch of images) |
| `frame_count` | INT | 16 | Max frames to sample from video |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `RESPONSE` | STRING | Model's text response (thinking stripped) |
| `THINKING` | STRING | Extracted reasoning content (empty if thinking disabled) |

### Recommended Sampling Parameters

From the [Qwen3.5 README](https://huggingface.co/Qwen/Qwen3.5-9B):

| Mode | Temperature | Top-p | Top-k | Repetition Penalty |
|------|-------------|-------|-------|---------------------|
| **Thinking** | 1.0 | 0.95 | 20 | 1.0 |
| **Instruct** (default) | 0.7 | 0.8 | 20 | 1.0 |

## How It Works

The node auto-detects the input mode:
- **Image provided** → image understanding
- **Video provided** → video understanding
- **Neither** → text-only generation

When thinking mode is enabled, the model generates internal reasoning (`<think>...</think>`) before the final response. The thinking tokens are automatically stripped from the output.

## Requirements

- `transformers >= 5.2.0`
- `torch`
- `bitsandbytes` (for quantization)
- `accelerate`
- CUDA GPU recommended

## License

Apache-2.0
