import os

# Disable transformers 5.x async weight loading to prevent OOM with ComfyUI's
# cudaMallocAsync allocator. Concurrent GPU allocations fragment memory pools.
# https://huggingface.co/docs/transformers/en/reference/environment_variables
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
