"""
Generation Pipeline for TRELLIS-HYBRID-V2 (Multi-View).

Pipeline: Image -> Qwen Multi-View (3 views) -> BiRefNet (each view) -> TRELLIS Multi-Image -> PLY

V2 Pipeline with multi-view generation using Qwen and multi-image Trellis.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional, List

from PIL import Image
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.birefnet_remover import BiRefNetBGRemover
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import secure_randint, set_random_seed, decode_image, to_png_base64, save_files, center_and_crop


# Multi-view prompts for Qwen
MULTIVIEW_PROMPTS = [
    "Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
    "Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
    "Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
]


class GenerationPipeline:
    """
    V2 Multi-View Pipeline.

    Flow:
    1. Input image -> Qwen generates 3 views (left, right, back)
    2. BiRefNet removes background from each view
    3. Multi-view images sent to Trellis (multidiffusion mode)

    Loading order:
    1. Trellis (first - takes most memory)
    2. Qwen
    3. BiRefNet
    """

    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize components (no VLM, no BEN2 - only BiRefNet)
        self.qwen_edit = QwenEditModule(settings)
        self.birefnet = BiRefNetBGRemover(device=f"cuda:{settings.qwen_gpu}")
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components in correct order."""
        logger.info("Starting V2 Multi-View pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Trellis first (takes most memory)
        await self.trellis.startup()
        self._clean_gpu_memory()

        # 2. Load Qwen
        await self.qwen_edit.startup()
        self._clean_gpu_memory()

        # 3. Load BiRefNet
        self.birefnet.load_model()
        self._clean_gpu_memory()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("V2 Multi-View Pipeline ready!")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing V2 Multi-View pipeline")

        self.birefnet.unload_model()
        await self.qwen_edit.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """Clean GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()

    def is_ready(self) -> bool:
        """Check if pipeline is ready."""
        return self.trellis.is_ready()

    async def warmup_generator(self) -> None:
        """Warmup the generator."""
        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        try:
            await self.generate_from_upload(temp_image_bytes, seed=42)
        except Exception as e:
            logger.warning(f"Warmup failed (this is okay): {e}")

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """Generate 3D model from uploaded image."""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        request = GenerateRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )

        response = await self.generate_gs(request)

        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64

    def _generate_multiview(self, image: Image.Image, seed: int) -> List[Image.Image]:
        """
        Generate multiple views using Qwen.

        Args:
            image: Input image
            seed: Random seed

        Returns:
            List of 3 view images (left three-quarters, right three-quarters, back)
        """
        logger.info("Generating multi-view images with Qwen...")
        views = []

        for i, prompt in enumerate(MULTIVIEW_PROMPTS):
            view_names = ["left three-quarters", "right three-quarters", "back"]
            logger.info(f"  View {i+1}/3: {view_names[i]}")

            view = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt=prompt,
            )
            views.append(view)

        logger.success(f"Generated {len(views)} views")
        return views

    def _remove_backgrounds(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Remove backgrounds from all images using BiRefNet.

        Args:
            images: List of images

        Returns:
            List of images with backgrounds removed
        """
        logger.info(f"Removing backgrounds from {len(images)} images...")
        result = []

        for i, img in enumerate(images):
            logger.info(f"  Processing image {i+1}/{len(images)}")
            img_no_bg = self.birefnet.remove_bg(img)
            result.append(img_no_bg)

        logger.success("Background removal complete")
        return result

    def _center_and_crop_all(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Center and crop all images.

        Args:
            images: List of images with transparent backgrounds

        Returns:
            List of centered and cropped images
        """
        logger.info(f"Centering and cropping {len(images)} images...")
        result = []

        for i, img in enumerate(images):
            cropped = center_and_crop(
                img,
                output_size=self.settings.output_image_size,
                padding_percentage=self.settings.padding_percentage,
                limit_padding=self.settings.limit_padding
            )
            result.append(cropped)

        logger.success(f"All images centered and cropped to {self.settings.output_image_size}")
        return result

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """Execute full V2 Multi-View generation pipeline."""
        t1 = time.time()
        logger.info("New V2 Multi-View generation request")

        # Set random seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Generate multi-view images with Qwen
        multiview_images = self._generate_multiview(image, request.seed)

        # 2. Remove backgrounds from all views
        multiview_no_bg = self._remove_backgrounds(multiview_images)

        # 3. Center and crop all images
        multiview_processed = self._center_and_crop_all(multiview_no_bg)

        # 4. Generate 3D model with multi-image Trellis
        trellis_params: TrellisParams = request.trellis_params

        trellis_result = self.trellis.generate_multi_image(
            TrellisRequest(
                images=multiview_processed,
                seed=request.seed,
                params=trellis_params
            )
        )

        # Save files if configured
        if self.settings.save_generated_files:
            save_files(
                trellis_result,
                image,
                multiview_images[0],  # First Qwen output
                multiview_no_bg[0],   # First BG removed
            )

        # Prepare response images
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(multiview_images[0])
            image_without_background_base64 = to_png_base64(multiview_processed[0])

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time:.2f}s")
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64 if self.settings.send_generated_files else None,
            image_without_background_file_base64=image_without_background_base64 if self.settings.send_generated_files else None,
        )
        return response
