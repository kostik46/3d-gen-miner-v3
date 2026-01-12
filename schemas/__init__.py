"""
Schemas for TRELLIS2-GS-GENERATOR.
Based on 3d-gen-miner-v12.
"""

from .overridable import OverridableModel
from .trellis_schemas import TrellisParams, TrellisParamsOverrides, TrellisRequest, TrellisResult
from .requests import GenerateRequest
from .responses import GenerateResponse, HealthResponse

__all__ = [
    "OverridableModel",
    "TrellisParams",
    "TrellisParamsOverrides",
    "TrellisRequest",
    "TrellisResult",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
]
