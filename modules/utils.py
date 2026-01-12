"""
Utility functions for TRELLIS2-GS-GENERATOR.
Based on 3d-gen-miner-v12.
"""

import base64
import io
import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from logger_config import logger
from config import settings


def center_and_crop(
    image: Image.Image,
    output_size: Tuple[int, int] = (518, 518),
    padding_percentage: float = 0.2,
    limit_padding: bool = True
) -> Image.Image:
    """
    Center and crop RGBA image around the object (non-transparent area).
    Based on 3d-gen-miner-v12.

    Args:
        image: RGBA PIL Image with alpha channel
        output_size: Target output size (width, height)
        padding_percentage: Padding around the object (0.2 = 20%)
        limit_padding: Whether to limit crop to image bounds

    Returns:
        Cropped and resized RGBA image
    """
    if image.mode != 'RGBA':
        logger.warning(f"center_and_crop: expected RGBA, got {image.mode}")
        image = image.convert('RGBA')

    img_array = np.array(image)
    alpha = img_array[:, :, 3]

    # Find bounding box of object
    bbox_indices = np.argwhere(alpha > 0.8 * 255)

    if len(bbox_indices) == 0:
        logger.warning("center_and_crop: no object found in alpha channel")
        return image.resize(output_size, Image.Resampling.LANCZOS)

    # Get bounding box
    y_min, x_min = bbox_indices.min(axis=0)
    y_max, x_max = bbox_indices.max(axis=0)

    # Calculate center and size
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)

    # Add padding
    padded_size = int(size * (1 + padding_percentage))

    # Calculate crop box
    left = int(center_x - padded_size // 2)
    top = int(center_y - padded_size // 2)
    right = int(center_x + padded_size // 2)
    bottom = int(center_y + padded_size // 2)

    if limit_padding:
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)
    else:
        # Expand canvas if needed
        if left < 0 or top < 0 or right > image.width or bottom > image.height:
            new_width = right - left
            new_height = bottom - top
            new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            paste_x = -left if left < 0 else 0
            paste_y = -top if top < 0 else 0
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
            left = max(0, left)
            top = max(0, top)
            right = left + new_width
            bottom = top + new_height

    # Crop and resize
    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize(output_size, Image.Resampling.LANCZOS)

    return resized


def secure_randint(low: int, high: int) -> int:
    """Return a random integer in [low, high] using os.urandom."""
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2 ** (8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)


def set_random_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_image(prompt: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def to_png_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def premultiply_alpha(image: Image.Image) -> Image.Image:
    """
    Premultiply alpha channel for RGBA image.
    Returns RGB image with alpha-premultiplied colors.
    Based on 3d-gen-miner-v12 trellis_manager.py.
    """
    if image.mode == 'RGBA':
        img_array = np.array(image).astype(np.float32) / 255.0
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4]
        premultiplied = rgb * alpha
        return Image.fromarray((premultiplied * 255).astype(np.uint8))
    else:
        return image.convert("RGB")


def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")


def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")


def save_files(
    trellis_result,
    image_edited: Image.Image,
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save PLY file
    if trellis_result:
        if trellis_result.ply_file:
            save_file_bytes(trellis_result.ply_file, "ply", "mesh", suffix=".ply")

    # Save images
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)
