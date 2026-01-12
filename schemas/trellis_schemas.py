"""
Trellis schemas based on 3d-gen-miner-v12.
Updated for V2 Multi-View support.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from PIL import Image

from .overridable import OverridableModel


class TrellisParams(OverridableModel):
    """Trellis parameters with automatic fallback to settings."""
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    slat_steps: int
    slat_cfg_strength: float
    num_oversamples: int = 1

    @classmethod
    def from_settings(cls, settings) -> "TrellisParams":
        return cls(
            sparse_structure_steps=settings.sparse_structure_steps,
            sparse_structure_cfg_strength=settings.sparse_structure_cfg,
            slat_steps=settings.slat_steps,
            slat_cfg_strength=settings.slat_cfg,
            num_oversamples=settings.num_oversamples,
        )


TrellisParamsOverrides = TrellisParams.Overrides


@dataclass
class TrellisRequest:
    """
    Request for Trellis 3D generation (internal use only).

    Supports both single image and multi-image (V2 multi-view).
    Use either `image` for single image or `images` for multi-view.
    """
    seed: int
    image: Optional[Image.Image] = None
    images: Optional[List[Image.Image]] = None
    params: Optional[TrellisParamsOverrides] = None


@dataclass(slots=True)
class TrellisResult:
    """Result from Trellis 3D generation."""
    ply_file: bytes | None = None
