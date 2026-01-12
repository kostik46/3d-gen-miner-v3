"""
SC-VAEs (Sparse Convolution VAEs) for TRELLIS.2
"""

from .sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
# Note: fdg_vae requires o_voxel which is not included
# from .fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder

__all__ = [
    'SparseUnetVaeEncoder',
    'SparseUnetVaeDecoder',
]

