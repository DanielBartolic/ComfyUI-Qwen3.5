# ComfyUI Load Image from URL node
# Downloads an image from a URL, outputs IMAGE tensor + URL string.
# Shows preview in the node after execution (like PreviewImage).

import os
import io
import uuid
import urllib.request

import numpy as np
import torch
from PIL import Image

import folder_paths


class LoadImageFromURL:
    """Load an image from a URL. Outputs IMAGE tensor and passes through the URL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Image URL (Pinterest, R2, any public URL)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "URL")
    FUNCTION = "load"
    CATEGORY = "Qwen3.5"
    OUTPUT_NODE = True

    def load(self, url: str):
        url = url.strip()
        if not url:
            raise ValueError("URL is empty")

        print(f"[Load Image URL] Downloading: {url[:100]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        img = Image.open(io.BytesIO(data)).convert("RGB")
        print(f"[Load Image URL] Loaded {img.size[0]}x{img.size[1]}")

        # Convert to ComfyUI IMAGE tensor (B, H, W, C) float32 0-1
        array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)

        # Save to temp dir for preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"url_preview_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(temp_dir, filename)
        img.save(filepath)

        return {
            "ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]},
            "result": (tensor, url),
        }


NODE_CLASS_MAPPINGS = {"LoadImageFromURL": LoadImageFromURL}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageFromURL": "Load Image from URL"}
