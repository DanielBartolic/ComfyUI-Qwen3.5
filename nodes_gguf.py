# ComfyUI-Qwen3.5 GGUF
# Fast inference node using llama.cpp (via llama-mtmd-cli subprocess).
# 9x faster than transformers FP16: 152 tok/s vs 17 tok/s on RTX PRO 6000.
#
# Requires: llama.cpp built with CUDA (llama-mtmd-cli binary on PATH or cli_path set)
# Models: https://huggingface.co/unsloth

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import folder_paths

MODELS = {
    "Qwen3.5-0.8B": "unsloth/Qwen3.5-0.8B-GGUF",
    "Qwen3.5-2B": "unsloth/Qwen3.5-2B-GGUF",
    "Qwen3.5-4B": "unsloth/Qwen3.5-4B-GGUF",
    "Qwen3.5-9B": "unsloth/Qwen3.5-9B-GGUF",
    "Qwen3.5-27B": "unsloth/Qwen3.5-27B-GGUF",
}
MODEL_OPTIONS = list(MODELS.keys())

QUANTIZATIONS = [
    "Q4_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_XL",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q6_K_XL",
    "Q8_0",
    "Q8_K_XL",
    "Q3_K_M",
    "Q3_K_S",
    "Q4_0",
    "Q4_1",
    "IQ4_NL",
    "IQ4_XS",
    "BF16",
]

# Map quantization names to GGUF filename patterns.
# "UD-" prefix is used for Unsloth Dynamic quantizations (_XL variants).
_UD_QUANTS = {"Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL", "Q8_K_XL"}

MMPROJ_FILENAME = "mmproj-BF16.gguf"

THINK_BLOCK_RE = re.compile(
    r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL
)


class Qwen35GGUF:
    """Qwen3.5 GGUF node — fast inference via llama.cpp."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {
                    "default": "Qwen3.5-9B",
                    "tooltip": "Model size. 0.8B ~1GB, 2B ~2GB, 4B ~3GB, 9B ~6GB, 27B ~17GB (Q4)",
                }),
                "quantization": (QUANTIZATIONS, {
                    "default": "Q4_K_XL",
                    "tooltip": "GGUF quantization. XL = Unsloth Dynamic (smart mixed precision)",
                }),
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
                    "max": 32768,
                    "tooltip": "Maximum tokens to generate",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0.6-0.7 recommended for captioning)",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold",
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Top-K sampling",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeated tokens",
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99,
                    "min": -1,
                    "max": 200,
                    "tooltip": "-1 or 99 offloads all layers to GPU",
                }),
                "ctx_size": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 131072,
                    "step": 1024,
                    "tooltip": "Context window size in tokens",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking mode. Outputs reasoning in THINKING output.",
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image for vision tasks"}),
                "cli_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to llama-mtmd-cli binary. Auto-detected if empty.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING")
    FUNCTION = "process"
    CATEGORY = "Qwen3.5"

    @staticmethod
    def _get_model_dir(model_name: str) -> Path:
        model_dir = Path(folder_paths.models_dir) / "LLM" / f"{model_name}-GGUF"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def _gguf_filename(model_name: str, quantization: str) -> str:
        """Build the GGUF filename from model name and quantization."""
        prefix = "UD-" if quantization in _UD_QUANTS else ""
        return f"{model_name}-{prefix}{quantization}.gguf"

    @staticmethod
    def _ensure_model(model_name: str, quantization: str) -> tuple[Path, Path]:
        """Download GGUF model + mmproj if not present."""
        repo_id = MODELS[model_name]
        model_dir = Qwen35GGUF._get_model_dir(model_name)
        model_filename = Qwen35GGUF._gguf_filename(model_name, quantization)
        model_path = model_dir / model_filename
        mmproj_path = model_dir / MMPROJ_FILENAME

        for filename, path in [
            (model_filename, model_path),
            (MMPROJ_FILENAME, mmproj_path),
        ]:
            if not path.exists():
                print(f"[Qwen3.5 GGUF] Downloading {filename} from {repo_id}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(model_dir),
                )
                print(f"[Qwen3.5 GGUF] Downloaded {filename}")

        return model_path, mmproj_path

    @staticmethod
    def _find_cli(cli_path_override: str) -> str:
        """Find the llama-mtmd-cli binary."""
        if cli_path_override and cli_path_override.strip():
            p = cli_path_override.strip()
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
            raise FileNotFoundError(
                f"[Qwen3.5 GGUF] llama-mtmd-cli not found at: {p}"
            )

        # Check PATH first
        found = shutil.which("llama-mtmd-cli")
        if found:
            return found

        # Check common locations
        candidates = [
            "/usr/local/bin/llama-mtmd-cli",
            "/opt/llama.cpp/build/bin/llama-mtmd-cli",
            "/workspace/llama.cpp/build/bin/llama-mtmd-cli",
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c

        raise FileNotFoundError(
            "[Qwen3.5 GGUF] llama-mtmd-cli not found. "
            "Build llama.cpp with CUDA or set the cli_path input."
        )

    @staticmethod
    def _tensor_to_temp_image(tensor: torch.Tensor) -> str:
        """Save ComfyUI IMAGE tensor as a temporary PNG. Returns file path."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil.save(path, format="PNG")
        return path

    @staticmethod
    def _invoke_cli(
        cli_path: str,
        model_path: Path,
        mmproj_path: Path,
        prompt: str,
        system_prompt: str,
        image_path: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        n_gpu_layers: int,
        ctx_size: int,
        enable_thinking: bool,
        seed: int,
    ) -> str:
        """Run llama-mtmd-cli and return the generated text."""
        thinking_kwargs = json.dumps({"enable_thinking": enable_thinking})
        cmd = [
            cli_path,
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--repeat-penalty", str(repeat_penalty),
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "--seed", str(seed),
            "--chat-template-kwargs", thinking_kwargs,
        ]

        if image_path:
            cmd.extend(["--image", image_path])

        # Build the full prompt with system prompt if provided
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"
        else:
            full_prompt = prompt

        cmd.extend(["-p", full_prompt])

        print(f"[Qwen3.5 GGUF] Running inference ({model_path.name})...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Filter out common warning lines
            error_lines = [
                line for line in stderr.split("\n")
                if not line.startswith(("ggml_", "llama_", "load_", "print_info",
                                        "common_init", "sched_", "clip_", "warmup",
                                        "main:", "WARN:", "find_slot"))
                and line.strip()
            ]
            error_msg = "\n".join(error_lines) if error_lines else stderr[-500:]
            raise RuntimeError(
                f"[Qwen3.5 GGUF] Inference failed (exit {result.returncode}): {error_msg}"
            )

        return result.stdout

    @staticmethod
    def _extract_thinking(text: str) -> tuple[str, str]:
        """Extract thinking content and clean response. Returns (response, thinking)."""
        thinking = ""

        # Case 1: Complete <think>...</think> block
        match = THINK_BLOCK_RE.search(text)
        if match:
            thinking = re.sub(r"</?think[^>]*>", "", match.group(0)).strip()
            text = THINK_BLOCK_RE.sub("", text).strip()

        # Case 2: </think> without opening tag (stripped by tokenizer)
        elif "</think>" in text:
            parts = text.split("</think>", 1)
            thinking = parts[0].strip()
            text = parts[1].strip()

        # Case 3: <think> without </think> (truncated by max_tokens)
        elif "<think>" in text:
            parts = text.split("<think>", 1)
            before = parts[0].strip()
            thinking = parts[1].strip()
            text = before

        # Clean leftover chat template tokens
        for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(token, "")
        return text.strip(), thinking

    def process(
        self,
        model: str,
        quantization: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        n_gpu_layers: int,
        ctx_size: int,
        enable_thinking: bool,
        seed: int,
        image=None,
        cli_path: str = "",
    ):
        cli = Qwen35GGUF._find_cli(cli_path)
        model_path, mmproj_path = Qwen35GGUF._ensure_model(model, quantization)

        image_path = None
        try:
            if image is not None:
                image_path = Qwen35GGUF._tensor_to_temp_image(image)

            raw_output = Qwen35GGUF._invoke_cli(
                cli_path=cli,
                model_path=model_path,
                mmproj_path=mmproj_path,
                prompt=prompt,
                system_prompt=system_prompt,
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                n_gpu_layers=n_gpu_layers,
                ctx_size=ctx_size,
                enable_thinking=enable_thinking,
                seed=seed,
            )

            response, thinking = Qwen35GGUF._extract_thinking(raw_output)

            if not enable_thinking:
                thinking = ""

            return (response, thinking)

        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)


NODE_CLASS_MAPPINGS = {"Qwen35GGUF": Qwen35GGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35GGUF": "Qwen 3.5 (GGUF)"}
