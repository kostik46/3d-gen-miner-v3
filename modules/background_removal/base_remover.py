from __future__ import annotations

from abc import ABC, abstractmethod
from PIL import Image


class BaseBGRemover(ABC):
    """Base class for background removal models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into GPU memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from GPU memory."""
        pass

    @abstractmethod
    def remove_bg(self, image: Image.Image) -> Image.Image:
        """Remove background from image."""
        pass
