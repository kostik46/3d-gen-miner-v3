"""
Qwen Image Edit Module using diffusers + LoRA.
Based on 3d-gen-miner-v12.
"""

import json
import math
import time
from os import PathLike
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel

from logger_config import logger
from config import Settings


class TextPrompting(BaseModel):
    prompt: str = Field(alias="positive")
    negative_prompt: Optional[str] = Field(default=None, alias="negative")


class QwenEditModule:
    """Qwen module for image editing using diffusers + LoRA. Based on v12."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = None
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"
        self.dtype = self._resolve_dtype(settings.qwen_dtype)
        self._empty_image = Image.new('RGB', (1024, 1024))

        # Load prompts from config file
        self.prompt_path = settings.qwen_edit_prompt_path
        self.prompting = self._load_prompting()

        # Pipe config
        self.pipe_config = {
            "num_inference_steps": settings.qwen_num_inference_steps,
            "true_cfg_scale": settings.qwen_true_cfg_scale,
            "height": settings.qwen_height,
            "width": settings.qwen_width,
        }

    def _load_prompting(self, path: Optional[PathLike] = None) -> TextPrompting:
        """Load prompting from JSON file."""
        path = Path(path or self.prompt_path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return TextPrompting.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load prompt from {path}: {e}, using defaults")
            return TextPrompting(
                positive="Show this object in three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
                negative="NSFW, (worst quality:2), (low quality:2)"
            )

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    def _get_scheduler_config(self) -> dict:
        """Return scheduler configuration for image editing."""
        return {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

    async def startup(self) -> None:
        """Initialize the Qwen pipeline."""
        logger.info("Initializing QwenEditModule...")
        await self._load_pipeline()
        logger.success("QwenEditModule ready.")

    async def _load_pipeline(self) -> None:
        """Load the complete pipeline."""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.settings.qwen_gpu)
            except Exception as err:
                logger.warning(f"Failed to set CUDA device: {err}")

        t1 = time.time()

        # Load transformer
        transformer = QwenImageTransformer2DModel.from_pretrained(
            self.settings.qwen_model_path,
            subfolder="transformer",
            torch_dtype=self.dtype
        )

        # Create scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._get_scheduler_config())

        # Create pipeline
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.settings.qwen_model_path,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=self.dtype
        )

        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {self.settings.qwen_lora_path}...")
        self.pipe.load_lora_weights(
            self.settings.qwen_lora_path,
            weight_name=self.settings.qwen_lora_filename
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        load_time = time.time() - t1
        logger.success(f"Qwen pipeline ready ({load_time:.2f}s). Device: {self.device}, dtype: {self.dtype}")

    async def shutdown(self) -> None:
        """Shutdown and free resources."""
        if self.pipe:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
        self.pipe = None
        logger.info("QwenEditModule closed.")

    def is_ready(self) -> bool:
        """Check if pipeline is loaded."""
        return self.pipe is not None

    def _prepare_input_image(self, image: Image.Image, megapixels: float = 1.0) -> Image.Image:
        """Prepare input image with proper size."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        total = int(megapixels * 1024 * 1024)
        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def edit_image(
        self,
        prompt_image: Image.Image,
        seed: int,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Edit image using Qwen.
        Based on 3d-gen-miner-v12.

        Args:
            prompt_image: Input image to edit
            seed: Random seed
            prompt: Custom prompt (overrides default if provided)
            negative_prompt: Custom negative prompt (overrides default if provided)

        Returns:
            Edited image
        """
        if self.pipe is None:
            raise RuntimeError("Qwen pipeline not loaded. Call startup() first.")

        start_time = time.time()

        # Prepare image
        image = self._prepare_input_image(prompt_image)

        # Get prompts - use custom if provided, otherwise use defaults
        prompting = {
            "prompt": prompt if prompt else self.prompting.prompt,
            "negative_prompt": negative_prompt if negative_prompt else self.prompting.negative_prompt
        }

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            image=image,
            generator=generator,
            **prompting,
            **self.pipe_config
        )

        edited_image = result.images[0]
        generation_time = time.time() - start_time

        logger.success(f"Qwen edit completed in {generation_time:.2f}s, Size: {edited_image.size}, Seed: {seed}")

        return edited_image
