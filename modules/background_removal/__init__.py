"""
Background Removal module.
Uses BiRefNet for background removal.
"""

from .base_remover import BaseBGRemover
from .birefnet_remover import BiRefNetBGRemover

__all__ = ["BaseBGRemover", "BiRefNetBGRemover"]
