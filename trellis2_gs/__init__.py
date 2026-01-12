"""
TRELLIS-HYBRID-V2 Gaussian Splatting Generator
"""

__version__ = "2.0.0"

from .pipelines import HybridTrellisGSPipeline
from .representations import Gaussian

__all__ = [
    "HybridTrellisGSPipeline",
    "Gaussian",
]
