import os

# Disable transformers 5.x async weight loading to prevent OOM with ComfyUI's
# cudaMallocAsync allocator. Concurrent GPU allocations fragment memory pools.
# https://huggingface.co/docs/transformers/en/reference/environment_variables
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

from .nodes import NODE_CLASS_MAPPINGS as _HF, NODE_DISPLAY_NAME_MAPPINGS as _HF_NAMES

NODE_CLASS_MAPPINGS = dict(_HF)
NODE_DISPLAY_NAME_MAPPINGS = dict(_HF_NAMES)

try:
    from .nodes_gguf import NODE_CLASS_MAPPINGS as _GGUF, NODE_DISPLAY_NAME_MAPPINGS as _GGUF_NAMES
    NODE_CLASS_MAPPINGS.update(_GGUF)
    NODE_DISPLAY_NAME_MAPPINGS.update(_GGUF_NAMES)
except ImportError as e:
    print(f"[Qwen3.5] GGUF node not available: {e}")

try:
    from .nodes_wavespeed import NODE_CLASS_MAPPINGS as _WS, NODE_DISPLAY_NAME_MAPPINGS as _WS_NAMES
    NODE_CLASS_MAPPINGS.update(_WS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_WS_NAMES)
except ImportError as e:
    print(f"[Qwen3.5] WaveSpeed node not available: {e}")

try:
    from .nodes_load_url import NODE_CLASS_MAPPINGS as _URL, NODE_DISPLAY_NAME_MAPPINGS as _URL_NAMES
    NODE_CLASS_MAPPINGS.update(_URL)
    NODE_DISPLAY_NAME_MAPPINGS.update(_URL_NAMES)
except ImportError as e:
    print(f"[Qwen3.5] Load URL node not available: {e}")

try:
    from .nodes_wavespeed_llm import NODE_CLASS_MAPPINGS as _LLM, NODE_DISPLAY_NAME_MAPPINGS as _LLM_NAMES
    NODE_CLASS_MAPPINGS.update(_LLM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_LLM_NAMES)
except ImportError as e:
    print(f"[Qwen3.5] WaveSpeed LLM node not available: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
