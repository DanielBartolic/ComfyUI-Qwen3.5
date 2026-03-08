# Universal WaveSpeed LLM API Node for ComfyUI
# OpenAI-compatible API supporting 199 models across 9 providers.
# No local GPU needed — runs on WaveSpeed's infrastructure.
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

# (display_name, model_id, supports_vision)
_MODELS = [
    # ── OpenAI ──
    ("OpenAI / GPT-5.3 Chat", "openai/gpt-5.3-chat", True),
    ("OpenAI / GPT-5.3 Codex", "openai/gpt-5.3-codex", True),
    ("OpenAI / GPT-5.2 Pro ($23/Mt)", "openai/gpt-5.2-pro", True),
    ("OpenAI / GPT-5.2 Chat", "openai/gpt-5.2-chat", True),
    ("OpenAI / GPT-5.2 Codex", "openai/gpt-5.2-codex", True),
    ("OpenAI / GPT-5.2", "openai/gpt-5.2", True),
    ("OpenAI / GPT-5.1", "openai/gpt-5.1", True),
    ("OpenAI / GPT-5.1 Chat", "openai/gpt-5.1-chat", True),
    ("OpenAI / GPT-5.1 Codex", "openai/gpt-5.1-codex", True),
    ("OpenAI / GPT-5.1 Codex Max", "openai/gpt-5.1-codex-max", True),
    ("OpenAI / GPT-5.1 Codex Mini", "openai/gpt-5.1-codex-mini", True),
    ("OpenAI / GPT-5", "openai/gpt-5", True),
    ("OpenAI / GPT-5 Chat", "openai/gpt-5-chat", True),
    ("OpenAI / GPT-5 Codex", "openai/gpt-5-codex", True),
    ("OpenAI / GPT-5 Mini", "openai/gpt-5-mini", True),
    ("OpenAI / GPT-5 Nano", "openai/gpt-5-nano", True),
    ("OpenAI / GPT-5 Image", "openai/gpt-5-image", True),
    ("OpenAI / GPT-5 Image Mini", "openai/gpt-5-image-mini", True),
    ("OpenAI / GPT-4.1", "openai/gpt-4.1", True),
    ("OpenAI / GPT-4.1 Mini", "openai/gpt-4.1-mini", True),
    ("OpenAI / GPT-4.1 Nano", "openai/gpt-4.1-nano", True),
    ("OpenAI / GPT-4o", "openai/gpt-4o", True),
    ("OpenAI / GPT-4o Extended", "openai/gpt-4o:extended", True),
    ("OpenAI / GPT-4o Mini", "openai/gpt-4o-mini", True),
    ("OpenAI / GPT-4o Audio Preview", "openai/gpt-4o-audio-preview", False),
    ("OpenAI / GPT-4 Turbo", "openai/gpt-4-turbo", True),
    ("OpenAI / GPT-4", "openai/gpt-4", False),
    ("OpenAI / GPT-3.5 Turbo", "openai/gpt-3.5-turbo", False),
    ("OpenAI / GPT Audio", "openai/gpt-audio", False),
    ("OpenAI / GPT Audio Mini", "openai/gpt-audio-mini", False),
    ("OpenAI / GPT-OSS 120B", "openai/gpt-oss-120b", False),
    ("OpenAI / GPT-OSS 120B Exacto", "openai/gpt-oss-120b:exacto", False),
    ("OpenAI / GPT-OSS 20B", "openai/gpt-oss-20b", False),
    ("OpenAI / o1", "openai/o1", True),
    ("OpenAI / o1 Pro ($165/Mt)", "openai/o1-pro", True),
    ("OpenAI / o3", "openai/o3", True),
    ("OpenAI / o3 Mini", "openai/o3-mini", False),
    ("OpenAI / o3 Deep Research", "openai/o3-deep-research", True),
    ("OpenAI / o4 Mini", "openai/o4-mini", True),
    ("OpenAI / o4 Mini High", "openai/o4-mini-high", True),
    ("OpenAI / o4 Mini Deep Research", "openai/o4-mini-deep-research", True),

    # ── Anthropic ──
    ("Anthropic / Claude Opus 4.6", "anthropic/claude-opus-4.6", True),
    ("Anthropic / Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6", True),
    ("Anthropic / Claude Opus 4.5", "anthropic/claude-opus-4.5", True),
    ("Anthropic / Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5", True),
    ("Anthropic / Claude Opus 4.1", "anthropic/claude-opus-4.1", True),
    ("Anthropic / Claude Opus 4", "anthropic/claude-opus-4", True),
    ("Anthropic / Claude Sonnet 4", "anthropic/claude-sonnet-4", True),
    ("Anthropic / Claude Haiku 4.5", "anthropic/claude-haiku-4.5", True),
    ("Anthropic / Claude 3.7 Sonnet", "anthropic/claude-3.7-sonnet", True),
    ("Anthropic / Claude 3.7 Sonnet Thinking", "anthropic/claude-3.7-sonnet:thinking", True),
    ("Anthropic / Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet", True),
    ("Anthropic / Claude 3.5 Haiku", "anthropic/claude-3.5-haiku", True),
    ("Anthropic / Claude 3 Haiku", "anthropic/claude-3-haiku", True),

    # ── Google ──
    ("Google / Gemini 3.1 Pro Preview", "google/gemini-3.1-pro-preview", True),
    ("Google / Gemini 3.1 Flash Image Preview", "google/gemini-3.1-flash-image-preview", True),
    ("Google / Gemini 3.1 Flash Lite Preview", "google/gemini-3.1-flash-lite-preview", True),
    ("Google / Gemini 3 Pro Preview", "google/gemini-3-pro-preview", True),
    ("Google / Gemini 3 Pro Image Preview", "google/gemini-3-pro-image-preview", True),
    ("Google / Gemini 3 Flash Preview", "google/gemini-3-flash-preview", True),
    ("Google / Gemini 2.5 Pro", "google/gemini-2.5-pro", True),
    ("Google / Gemini 2.5 Flash", "google/gemini-2.5-flash", True),
    ("Google / Gemini 2.5 Flash Image", "google/gemini-2.5-flash-image", True),
    ("Google / Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", True),
    ("Google / Gemini 2.0 Flash", "google/gemini-2.0-flash-001", True),
    ("Google / Gemini 2.0 Flash Lite", "google/gemini-2.0-flash-lite-001", True),
    ("Google / Gemma 3n E4B", "google/gemma-3n-e4b-it", False),
    ("Google / Gemma 3 4B", "google/gemma-3-4b-it", True),
    ("Google / Gemma 3 12B", "google/gemma-3-12b-it", True),
    ("Google / Gemma 3 27B", "google/gemma-3-27b-it", True),
    ("Google / Gemma 2 9B", "google/gemma-2-9b-it", False),
    ("Google / Gemma 2 27B", "google/gemma-2-27b-it", False),

    # ── Qwen / Alibaba ──
    ("Qwen / Qwen3.5-35B-A3B", "qwen/qwen3.5-35b-a3b", True),
    ("Qwen / Qwen3.5-27B", "qwen/qwen3.5-27b", True),
    ("Qwen / Qwen3.5-122B-A10B", "qwen/qwen3.5-122b-a10b", True),
    ("Qwen / Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b", True),
    ("Qwen / Qwen3.5-Flash", "qwen/qwen3.5-flash-02-23", True),
    ("Qwen / Qwen3-8B", "qwen/qwen3-8b", False),
    ("Qwen / Qwen3-14B", "qwen/qwen3-14b", False),
    ("Qwen / Qwen3-32B", "qwen/qwen3-32b", False),
    ("Qwen / Qwen3-30B-A3B", "qwen/qwen3-30b-a3b", False),
    ("Qwen / Qwen3-235B-A22B", "qwen/qwen3-235b-a22b", False),
    ("Qwen / Qwen3-235B-A22B 2507", "qwen/qwen3-235b-a22b-2507", False),
    ("Qwen / Qwen3-235B-A22B Thinking 2507", "qwen/qwen3-235b-a22b-thinking-2507", False),
    ("Qwen / Qwen3-Next-80B-A3B Instruct", "qwen/qwen3-next-80b-a3b-instruct", False),
    ("Qwen / Qwen3-Next-80B-A3B Thinking", "qwen/qwen3-next-80b-a3b-thinking", False),
    ("Qwen / Qwen3-Max", "qwen/qwen3-max", False),
    ("Qwen / Qwen3-Max Thinking", "qwen/qwen3-max-thinking", False),
    ("Qwen / Qwen3-VL-8B Instruct", "qwen/qwen3-vl-8b-instruct", True),
    ("Qwen / Qwen3-VL-8B Thinking", "qwen/qwen3-vl-8b-thinking", True),
    ("Qwen / Qwen3-VL-30B-A3B Instruct", "qwen/qwen3-vl-30b-a3b-instruct", True),
    ("Qwen / Qwen3-VL-30B-A3B Thinking", "qwen/qwen3-vl-30b-a3b-thinking", True),
    ("Qwen / Qwen3-VL-32B Instruct", "qwen/qwen3-vl-32b-instruct", True),
    ("Qwen / Qwen3-VL-235B-A22B Instruct", "qwen/qwen3-vl-235b-a22b-instruct", True),
    ("Qwen / Qwen3-VL-235B-A22B Thinking", "qwen/qwen3-vl-235b-a22b-thinking", True),
    ("Qwen / Qwen3-Coder", "qwen/qwen3-coder", False),
    ("Qwen / Qwen3-Coder Exacto", "qwen/qwen3-coder:exacto", False),
    ("Qwen / Qwen3-Coder Next", "qwen/qwen3-coder-next", False),
    ("Qwen / Qwen3-Coder-30B-A3B", "qwen/qwen3-coder-30b-a3b-instruct", False),
    ("Qwen / Qwen3-Coder Flash", "qwen/qwen3-coder-flash", False),
    ("Qwen / Qwen3-Coder Plus", "qwen/qwen3-coder-plus", False),
    ("Qwen / QwQ-32B", "qwen/qwq-32b", False),
    ("Qwen / Qwen-Max", "qwen/qwen-max", False),
    ("Qwen / Qwen-Plus", "qwen/qwen-plus", False),
    ("Qwen / Qwen-Turbo", "qwen/qwen-turbo", False),
    ("Qwen / Qwen-VL-Max", "qwen/qwen-vl-max", True),
    ("Qwen / Qwen-VL-Plus", "qwen/qwen-vl-plus", True),
    ("Qwen / Qwen2.5-72B Instruct", "qwen/qwen-2.5-72b-instruct", False),
    ("Qwen / Qwen2.5-VL-32B Instruct", "qwen/qwen2.5-vl-32b-instruct", True),

    # ── DeepSeek ──
    ("DeepSeek / V3.2", "deepseek/deepseek-v3.2", False),
    ("DeepSeek / V3.2 Speciale", "deepseek/deepseek-v3.2-speciale", False),
    ("DeepSeek / V3.1 Chat", "deepseek/deepseek-chat-v3.1", False),
    ("DeepSeek / V3.1 Terminus", "deepseek/deepseek-v3.1-terminus", False),
    ("DeepSeek / V3.1 Terminus Exacto", "deepseek/deepseek-v3.1-terminus:exacto", False),
    ("DeepSeek / V3 Chat 0324", "deepseek/deepseek-chat-v3-0324", False),
    ("DeepSeek / Chat", "deepseek/deepseek-chat", False),
    ("DeepSeek / R1", "deepseek/deepseek-r1", False),
    ("DeepSeek / R1 0528", "deepseek/deepseek-r1-0528", False),
    ("DeepSeek / R1 Distill Llama 70B", "deepseek/deepseek-r1-distill-llama-70b", False),
    ("DeepSeek / R1 Distill Qwen 32B", "deepseek/deepseek-r1-distill-qwen-32b", False),

    # ── Meta / Llama ──
    ("Meta / Llama 4 Maverick", "meta-llama/llama-4-maverick", True),
    ("Meta / Llama 4 Scout", "meta-llama/llama-4-scout", True),
    ("Meta / Llama 3.3 70B Instruct", "meta-llama/llama-3.3-70b-instruct", False),
    ("Meta / Llama 3.2 11B Vision", "meta-llama/llama-3.2-11b-vision-instruct", True),
    ("Meta / Llama 3.2 3B Instruct", "meta-llama/llama-3.2-3b-instruct", False),
    ("Meta / Llama 3.1 405B", "meta-llama/llama-3.1-405b", False),
    ("Meta / Llama 3.1 405B Instruct", "meta-llama/llama-3.1-405b-instruct", False),
    ("Meta / Llama 3.1 70B Instruct", "meta-llama/llama-3.1-70b-instruct", False),
    ("Meta / Llama 3.1 8B Instruct", "meta-llama/llama-3.1-8b-instruct", False),
    ("Meta / Llama 3 70B Instruct", "meta-llama/llama-3-70b-instruct", False),
    ("Meta / Llama 3 8B Instruct", "meta-llama/llama-3-8b-instruct", False),

    # ── Mistral ──
    ("Mistral / Large 2512", "mistralai/mistral-large-2512", True),
    ("Mistral / Large", "mistralai/mistral-large", False),
    ("Mistral / Medium 3.1", "mistralai/mistral-medium-3.1", True),
    ("Mistral / Medium 3", "mistralai/mistral-medium-3", True),
    ("Mistral / Small 3.2 24B", "mistralai/mistral-small-3.2-24b-instruct", True),
    ("Mistral / Small 24B 2501", "mistralai/mistral-small-24b-instruct-2501", False),
    ("Mistral / Small Creative", "mistralai/mistral-small-creative", False),
    ("Mistral / Saba", "mistralai/mistral-saba", False),
    ("Mistral / Nemo", "mistralai/mistral-nemo", False),
    ("Mistral / Devstral Medium", "mistralai/devstral-medium", False),
    ("Mistral / Devstral 2512", "mistralai/devstral-2512", False),
    ("Mistral / Devstral Small", "mistralai/devstral-small", False),
    ("Mistral / Codestral 2508", "mistralai/codestral-2508", False),
    ("Mistral / Ministral 14B 2512", "mistralai/ministral-14b-2512", True),
    ("Mistral / Ministral 8B 2512", "mistralai/ministral-8b-2512", True),
    ("Mistral / Ministral 3B 2512", "mistralai/ministral-3b-2512", True),
    ("Mistral / Pixtral Large", "mistralai/pixtral-large-2411", True),
    ("Mistral / Mixtral 8x22B", "mistralai/mixtral-8x22b-instruct", False),
    ("Mistral / Mixtral 8x7B", "mistralai/mixtral-8x7b-instruct", False),
    ("Mistral / Voxtral Small 24B", "mistralai/voxtral-small-24b-2507", False),

    # ── xAI / Grok ──
    ("xAI / Grok 4.1 Fast", "x-ai/grok-4.1-fast", True),
    ("xAI / Grok 4", "x-ai/grok-4", True),
    ("xAI / Grok 4 Fast", "x-ai/grok-4-fast", True),
    ("xAI / Grok 3", "x-ai/grok-3", False),
    ("xAI / Grok 3 Beta", "x-ai/grok-3-beta", False),
    ("xAI / Grok 3 Mini", "x-ai/grok-3-mini", False),
    ("xAI / Grok 3 Mini Beta", "x-ai/grok-3-mini-beta", False),
    ("xAI / Grok Code Fast", "x-ai/grok-code-fast-1", False),

    # ── ChatGLM / Z-AI ──
    ("ChatGLM / GLM-5", "z-ai/glm-5", False),
    ("ChatGLM / GLM-4.7", "z-ai/glm-4.7", False),
    ("ChatGLM / GLM-4.6", "z-ai/glm-4.6", False),
    ("ChatGLM / GLM-4.6 Exacto", "z-ai/glm-4.6:exacto", False),
    ("ChatGLM / GLM-4.6V", "z-ai/glm-4.6v", True),
    ("ChatGLM / GLM-4.5", "z-ai/glm-4.5", False),
    ("ChatGLM / GLM-4.5V", "z-ai/glm-4.5v", True),
    ("ChatGLM / GLM-4.5 Air", "z-ai/glm-4.5-air", False),
    ("ChatGLM / GLM-4 32B", "z-ai/glm-4-32b", False),
]

# Build lookup dicts
MODELS = {name: model_id for name, model_id, _ in _MODELS}
VISION_MODELS = {model_id for _, model_id, vision in _MODELS if vision}
MODEL_OPTIONS = list(MODELS.keys())


class WaveSpeedLLM:
    """Universal WaveSpeed LLM API — 199 models, no local GPU needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {
                    "default": "Anthropic / Claude Sonnet 4.6",
                    "tooltip": "LLM model via WaveSpeed API. All use OpenAI-compatible API.",
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
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0=deterministic, 2=creative)",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "WaveSpeed API key. Leave empty to use WAVESPEED_API_KEY env var.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image input (vision models only — ignored for text-only models)",
                }),
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Image URL sent directly to API. Preferred over image tensor.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING")
    FUNCTION = "process"
    CATEGORY = "WaveSpeed LLM"

    @staticmethod
    def _tensor_to_base64(tensor: torch.Tensor, max_side: int = 1024) -> str:
        """Convert ComfyUI IMAGE tensor to base64 data URI."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(array)
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            print(f"[WaveSpeed LLM] Resized image {w}x{h} -> {img.size[0]}x{img.size[1]}")
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
        has_vision = model_id in VISION_MODELS

        # Build messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # Check if image is provided for a non-vision model
        has_image = (image is not None or (image_url and image_url.strip()))
        if has_image and not has_vision:
            print(f"[WaveSpeed LLM] Warning: {model_id} does not support vision. Image input ignored.")

        user_content = []
        if has_image and has_vision:
            if image_url and image_url.strip():
                user_content.append({"type": "image_url", "image_url": {"url": image_url.strip()}})
                print(f"[WaveSpeed LLM] Using image URL: {image_url.strip()[:80]}...")
            elif image is not None:
                data_uri = self._tensor_to_base64(image)
                user_content.append({"type": "image_url", "image_url": {"url": data_uri}})
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        print(f"[WaveSpeed LLM] Calling {model_id}...")
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0
        print(f"[WaveSpeed LLM] {model_id} — {tokens} tokens generated")

        # Extract thinking blocks (DeepSeek R1, Qwen thinking, Claude thinking, etc.)
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


NODE_CLASS_MAPPINGS = {"WaveSpeedLLM": WaveSpeedLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"WaveSpeedLLM": "WaveSpeed LLM (199 Models)"}
