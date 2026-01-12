"""
Trellis Service for Gaussian Splatting generation.
Based on 3d-gen-miner-v12.

Uses HybridTrellisGSPipeline:
- TRELLIS.2 for sparse structure (better geometry)
- TRELLIS1 for SLat flow + GS decoder (native Gaussian output)
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config import Settings
from schemas import TrellisResult, TrellisRequest, TrellisParams
from logger_config import logger


class TrellisService:
    """
    Service for Trellis 3D generation.
    Based on 3d-gen-miner-v12.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(settings)

    async def startup(self) -> None:
        """Initialize the Trellis pipeline."""
        logger.info("Loading Hybrid Trellis pipeline...")

        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        # Import here to avoid circular imports
        from trellis2_gs.pipelines import HybridTrellisGSPipeline

        logger.info(f"  Sparse Structure: {self.settings.trellis2_model_id}")
        logger.info(f"  SLat + GS Decoder: {self.settings.trellis1_model_id}")

        self.pipeline = HybridTrellisGSPipeline.from_pretrained(
            trellis2_path=self.settings.trellis2_model_id,
            trellis1_path=self.settings.trellis1_model_id,
            # Don't use built-in BEN2 - we use separate module
            background_removal_model_id=self.settings.background_removal_model_id,
        )
        self.pipeline.cuda()

        logger.success("Hybrid Trellis pipeline ready.")

    async def shutdown(self) -> None:
        """Shutdown the pipeline."""
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        """Check if pipeline is ready."""
        return self.pipeline is not None

    def generate(self, trellis_request: TrellisRequest) -> TrellisResult:
        """
        Generate 3D Gaussian Splatting from image.
        Based on 3d-gen-miner-v12.

        Handles RGBA → RGB conversion with alpha premultiplication.
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        image = trellis_request.image

        # RGBA → RGB with alpha premultiplication (from v12)
        if image.mode == 'RGBA':
            img_array = np.array(image).astype(np.float32) / 255.0
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3:4]
            premultiplied = rgb * alpha
            image_rgb = Image.fromarray((premultiplied * 255).astype(np.uint8))
        else:
            image_rgb = image.convert("RGB")

        logger.info(f"Generating Trellis seed={trellis_request.seed}, image size={image_rgb.size}")

        # Get params with overrides
        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        buffer = None

        try:
            outputs = self.pipeline.run(
                image_rgb,
                seed=trellis_request.seed,
                num_samples=1,  # Final output count
                num_oversamples=params.num_oversamples,  # Generate N structures, pick smallest
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,  # We already preprocessed!
                formats=["gaussian"],
            )

            generation_time = time.time() - start

            # Get gaussians list (pipeline already selected smallest by voxel count)
            gaussians = outputs.get("gaussian", [])
            if not gaussians:
                raise RuntimeError("No gaussians generated")

            gaussian = gaussians[0]

            # Save to PLY
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            num_gaussians = len(gaussian._xyz) if hasattr(gaussian, '_xyz') else 0

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None
            )

            logger.success(f"Trellis finished in {generation_time:.2f}s, {num_gaussians} gaussians")
            return result

        finally:
            if buffer:
                buffer.close()

    def generate_multi_image(self, trellis_request: TrellisRequest) -> TrellisResult:
        """
        Generate 3D Gaussian Splatting from multiple images (multi-view).

        Uses multidiffusion mode for combining multiple views.
        Based on 3d-gen-miner-v5.

        Args:
            trellis_request: Request with images list

        Returns:
            TrellisResult with PLY data
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images = trellis_request.images
        if not images or len(images) == 0:
            raise ValueError("No images provided for multi-image generation")

        # Convert all RGBA images to RGB with alpha premultiplication
        images_rgb = []
        for img in images:
            if img.mode == 'RGBA':
                img_array = np.array(img).astype(np.float32) / 255.0
                rgb = img_array[:, :, :3]
                alpha = img_array[:, :, 3:4]
                premultiplied = rgb * alpha
                images_rgb.append(Image.fromarray((premultiplied * 255).astype(np.uint8)))
            else:
                images_rgb.append(img.convert("RGB"))

        logger.info(f"Generating Trellis multi-image with {len(images_rgb)} views, seed={trellis_request.seed}")

        # Get params with overrides
        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        buffer = None

        try:
            outputs = self.pipeline.run_multi_image(
                images_rgb,
                seed=trellis_request.seed,
                num_samples=1,
                num_oversamples=params.num_oversamples,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,
                formats=["gaussian"],
                mode="multidiffusion",  # Combine all views at each step
            )

            generation_time = time.time() - start

            # Get gaussians list
            gaussians = outputs.get("gaussian", [])
            if not gaussians:
                raise RuntimeError("No gaussians generated")

            gaussian = gaussians[0]

            # Save to PLY
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            num_gaussians = len(gaussian._xyz) if hasattr(gaussian, '_xyz') else 0

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None
            )

            logger.success(f"Trellis multi-image finished in {generation_time:.2f}s, {num_gaussians} gaussians")
            return result

        finally:
            if buffer:
                buffer.close()
