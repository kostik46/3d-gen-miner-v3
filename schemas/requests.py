"""
Request schemas for TRELLIS2-GS-GENERATOR.
Based on 3d-gen-miner-v12.
"""

from typing import Optional

from pydantic import BaseModel, Field

from .trellis_schemas import TrellisParamsOverrides


class GenerateRequest(BaseModel):
    """Request body for 3D generation from image."""

    prompt_type: str = Field(
        default="image",
        description="Type of prompt (always 'image' for now)"
    )

    prompt_image: str = Field(
        ...,
        description="Base64 encoded input image"
    )

    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed (-1 for random)"
    )

    trellis_params: Optional[TrellisParamsOverrides] = Field(
        default=None,
        description="Optional TRELLIS generation parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt_type": "image",
                "prompt_image": "<base64_encoded_image>",
                "seed": 42,
                "trellis_params": {
                    "sparse_structure_steps": 8,
                    "sparse_structure_cfg_strength": 5.75,
                    "slat_steps": 20,
                    "slat_cfg_strength": 2.4,
                    "num_oversamples": 3
                }
            }
        }
