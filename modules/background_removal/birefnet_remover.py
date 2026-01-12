from __future__ import annotations

import gc
from PIL import Image

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from modules.background_removal.base_remover import BaseBGRemover
from logger_config import logger


class BiRefNetBGRemover(BaseBGRemover):
    """BiRefNet background remover (ZhengPeng7/BiRefNet)."""

    def __init__(self, device: str = "cuda:0"):
        self._bg_remover: AutoModelForImageSegmentation | None = None
        self._device = device
        torch.set_float32_matmul_precision('high')
        self._transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self) -> None:
        """Load BiRefNet model."""
        logger.info("Loading BiRefNet model...")
        self._bg_remover = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True
        )
        self._bg_remover.to(self._device)
        self._bg_remover.eval()
        self._bg_remover.half()
        logger.success("BiRefNet model loaded.")

    def unload_model(self) -> None:
        """Unload BiRefNet model."""
        if self._bg_remover is not None:
            del self._bg_remover
            gc.collect()
            torch.cuda.empty_cache()
            self._bg_remover = None
            logger.info("BiRefNet model unloaded.")

    def remove_bg(self, image: Image.Image) -> Image.Image:
        """Remove background using BiRefNet."""
        if self._bg_remover is None:
            raise RuntimeError("BiRefNet model not loaded.")

        # Convert to RGB if needed
        if image.mode != "RGB":
            rgb_image = image.convert("RGB")
        else:
            rgb_image = image.copy()

        # Transform and run inference
        input_tensor = self._transform_image(rgb_image).unsqueeze(0).to(self._device).half()

        with torch.no_grad():
            preds = self._bg_remover(input_tensor)[-1].sigmoid().cpu()

        # Create mask
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(rgb_image.size)

        # Apply mask to create RGBA image
        result_rgba = rgb_image.copy()
        result_rgba.putalpha(mask)

        return result_rgba
