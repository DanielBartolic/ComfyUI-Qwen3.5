# ComfyUI-Qwen3.5 WaveSpeed API Node
# Calls Qwen3.5 via WaveSpeed's OpenAI-compatible LLM API.
# No local GPU needed for captioning — runs on WaveSpeed's infrastructure.
#
# Requires: pip install openai
# API key: set WAVESPEED_API_KEY environment variable

import io
import os
import re
import base64

import numpy as np
import torch
from PIL import Image

THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

MODELS = {
    "Qwen3.5-35B-A3B (cheapest)": "qwen/qwen3.5-35b-a3b",
    "Qwen3.5-Flash": "qwen/qwen3.5-flash-02-23",
    "Qwen3.5-27B": "qwen/qwen3.5-27b",
    "Qwen3.5-Plus": "qwen/qwen3.5-plus-02-15",
    "Qwen3.5-122B-A10B": "qwen/qwen3.5-122b-a10b",
    "Qwen3.5-397B-A17B (best)": "qwen/qwen3.5-397b-a17b",
}
MODEL_OPTIONS = list(MODELS.keys())


class Qwen35WaveSpeed:
    """Qwen3.5 via WaveSpeed API — no local GPU needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {
                    "default": "Qwen3.5-27B",
                    "tooltip": (
                        "WaveSpeed model. 35B-A3B: $0.16/$1.30 per M tokens. "
                        "27B: $0.20/$1.60. 122B-A10B: $0.26/$2.10. "
                        "397B-A17B: $0.39/$2.30 (best quality)."
                    ),
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "Text prompt for the model",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system prompt",
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 81920,
                    "tooltip": "Maximum tokens to generate",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. Thinking: 1.0, instruct: 0.7",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling. Thinking: 0.95, instruct: 0.8",
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-K sampling",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "WaveSpeed API key. Leave empty to use WAVESPEED_API_KEY env var.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Single image input (resized + base64 encoded)"}),
                "image_url": ("STRING", {"default": "", "tooltip": "Image URL — sent directly to API, no base64. Preferred over image input."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING")
    FUNCTION = "process"
    CATEGORY = "Qwen3.5"

    @staticmethod
    def _tensor_to_base64(tensor: torch.Tensor, max_side: int = 1024) -> str:
        """Convert ComfyUI IMAGE tensor to base64 data URI, resized to fit API limits."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(array)
        # Resize if too large (keeps aspect ratio)
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            print(f"[Qwen3.5 WaveSpeed] Resized image {w}x{h} -> {img.size[0]}x{img.size[1]}")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def process(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        api_key: str,
        image=None,
        image_url="",
    ):
        from openai import OpenAI

        key = api_key.strip() or os.environ.get("WAVESPEED_API_KEY", "")
        if not key:
            raise RuntimeError(
                "WaveSpeed API key not set. Either pass it in the node or "
                "set WAVESPEED_API_KEY environment variable."
            )

        client = OpenAI(api_key=key, base_url="https://llm.wavespeed.ai/v1")
        model_id = MODELS[model]

        # Build messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        user_content = []
        # Prefer URL over base64 tensor (URL is sent directly, no size limit)
        if image_url and image_url.strip():
            user_content.append({"type": "image_url", "image_url": {"url": image_url.strip()}})
            print(f"[Qwen3.5 WaveSpeed] Using image URL: {image_url.strip()[:80]}...")
        elif image is not None:
            data_uri = self._tensor_to_base64(image)
            user_content.append({"type": "image_url", "image_url": {"url": data_uri}})
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        print(f"[Qwen3.5 WaveSpeed] Calling {model_id}...")
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={"top_k": top_k},
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0
        print(f"[Qwen3.5 WaveSpeed] Generated {tokens} tokens")

        # Extract thinking
        thinking = ""
        match = THINK_BLOCK_RE.search(text)
        if match:
            thinking = re.sub(r"</?think[^>]*>", "", match.group(0)).strip()
            text = THINK_BLOCK_RE.sub("", text).strip()
        if "</think>" in text:
            parts = text.split("</think>", 1)
            if not thinking:
                thinking = parts[0].strip()
            text = parts[1].strip()

        return (text, thinking)


NODE_CLASS_MAPPINGS = {"Qwen35WaveSpeed": Qwen35WaveSpeed}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35WaveSpeed": "Qwen 3.5 (WaveSpeed API)"}
