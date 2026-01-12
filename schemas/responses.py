"""
Response schemas for TRELLIS2-GS-GENERATOR.
Based on 3d-gen-miner-v12.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    """Response body for 3D generation."""

    generation_time: float = Field(
        ...,
        description="Total generation time in seconds"
    )

    ply_file_base64: Optional[str | bytes] = Field(
        default=None,
        description="Generated PLY file (bytes or base64 encoded)"
    )

    image_edited_file_base64: Optional[str] = Field(
        default=None,
        description="Qwen-edited image (base64 PNG)"
    )

    image_without_background_file_base64: Optional[str] = Field(
        default=None,
        description="Final preprocessed image (base64 PNG)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.5,
                "ply_file_base64": "<base64_encoded_ply>"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ready"
    pipeline_loaded: bool = False
    warmed_up: bool = False
