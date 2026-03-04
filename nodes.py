# ComfyUI-Qwen3.5
# Custom node for Qwen3.5-9B unified natively multimodal model.
# Supports text-only, image, and video inputs.
#
# Model: https://huggingface.co/Qwen/Qwen3.5-9B
# License: Apache-2.0

import gc
import re
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

try:
    from transformers import AutoModelForImageTextToText as AutoVLModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLModel
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

import folder_paths

MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_DIR_NAME = "Qwen3.5-9B"
QUANTIZATION_OPTIONS = ["FP16", "8-bit", "4-bit"]
THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


class Qwen35:
    """Qwen3.5-9B unified multimodal node for ComfyUI."""

    model = None
    processor = None
    tokenizer = None
    current_signature = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "Text prompt for the model",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system prompt to set model behavior",
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
                    "tooltip": "Sampling temperature. Thinking mode recommends 1.0, instruct mode 0.7",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling. Thinking mode recommends 0.95, instruct mode 0.8",
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-K sampling. Recommended: 20",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeated tokens. Recommended: 1.0",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking mode. Model outputs <think>...</think> reasoning before response.",
                }),
                "quantization": (QUANTIZATION_OPTIONS, {
                    "default": "FP16",
                    "tooltip": "Model quantization. 4-bit needs ~7GB VRAM, 8-bit ~12GB, FP16 ~20GB",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM between runs for faster inference",
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Single image input"}),
                "video": ("IMAGE", {"tooltip": "Video frames input (batch of images)"}),
                "frame_count": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Maximum number of frames to sample from video",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/Qwen3.5"

    @staticmethod
    def _get_model_path():
        """Get the local model path, creating the directory if needed."""
        llm_dir = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(llm_dir, exist_ok=True)
        return os.path.join(llm_dir, MODEL_DIR_NAME)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a ComfyUI IMAGE tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def _clear(self):
        """Release model from memory."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, quantization: str, keep_model_loaded: bool):
        """Load or reuse the model based on current config."""
        signature = f"{MODEL_REPO}_{quantization}"

        if self.model is not None and self.current_signature == signature:
            return

        if self.model is not None:
            self._clear()

        model_path = self._get_model_path()

        # Download if not present
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"[Qwen3.5] Downloading {MODEL_REPO}...")
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=model_path,
                ignore_patterns=["*.gguf"],
            )

        # Build quantization config
        quant_config = None

        if quantization == "4-bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8-bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Reset any VRAM memory fraction limit set by ComfyUI
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(1.0)
            except Exception:
                pass

        # Use device_map="cuda" (NOT "auto") to avoid accelerate's parallel
        # tensor materialization which causes OOM inside ComfyUI.
        # This matches the pattern used by ComfyUI-QwenVL.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs = {
            "use_safetensors": True,
            "device_map": device,
        }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024 ** 3)
            print(f"[Qwen3.5] GPU memory: {free_gb:.1f} GiB free / {total_mem / (1024**3):.1f} GiB total")

        print(f"[Qwen3.5] Loading model ({quantization})...")
        self.model = AutoVLModel.from_pretrained(
            model_path,
            **load_kwargs,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature
        print("[Qwen3.5] Model loaded.")

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        system_prompt: str,
        image,
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        enable_thinking: bool,
    ) -> str:
        """Run inference with the loaded model."""
        # Build conversation
        messages = []

        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        user_content = []

        # Image mode
        if image is not None:
            pil_image = self._tensor_to_pil(image)
            user_content.append({"type": "image", "image": pil_image})

        # Video mode
        if video is not None:
            frames = [self._tensor_to_pil(frame) for frame in video]
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                user_content.append({"type": "video", "video": frames})

        # Text prompt (always present)
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # Apply chat template with thinking mode control
        chat = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

        # Process inputs
        images = [item["image"] for item in user_content if item.get("type") == "image"]
        video_frames = [
            frame
            for item in user_content
            if item.get("type") == "video"
            for frame in item["video"]
        ]
        videos = [video_frames] if video_frames else None

        processed = self.processor(
            text=chat,
            images=images or None,
            videos=videos,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }

        # Stop tokens
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        # Generate
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        outputs = self.model.generate(**model_inputs, **gen_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Decode — trim input tokens
        input_len = model_inputs["input_ids"].shape[-1]
        generated = outputs[0, input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Strip thinking blocks from output
        text = THINK_BLOCK_RE.sub("", text).strip()

        return text

    def process(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        enable_thinking: bool,
        quantization: str,
        keep_model_loaded: bool,
        seed: int,
        image=None,
        video=None,
        frame_count: int = 16,
    ):
        torch.manual_seed(seed)

        self._load_model(quantization, keep_model_loaded)

        try:
            text = self._generate(
                prompt=prompt,
                system_prompt=system_prompt,
                image=image,
                video=video,
                frame_count=frame_count,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self._clear()


NODE_CLASS_MAPPINGS = {"Qwen35": Qwen35}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35": "Qwen 3.5"}
