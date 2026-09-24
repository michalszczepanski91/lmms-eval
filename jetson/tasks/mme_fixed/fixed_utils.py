import os

from PIL import Image


def doc_to_visual(doc):
    """MME image resized to a FIXED_IMAGE_SIDE x FIXED_IMAGE_SIDE square (a multiple of 28 keeps it unchanged by Qwen2.5-VL)."""
    side = int(os.environ.get("FIXED_IMAGE_SIDE", "448"))
    return [doc["image"].convert("RGB").resize((side, side), Image.Resampling.BICUBIC)]
