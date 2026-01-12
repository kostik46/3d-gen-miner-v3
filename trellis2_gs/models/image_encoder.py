"""
Image encoders for conditioning the generation.
"""

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class ImageEncoder(nn.Module):
    """
    Base class for image encoders.
    """
    
    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def preprocess(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess image(s) for the encoder."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images."""
        raise NotImplementedError


class DINOv2Encoder(ImageEncoder):
    """
    DINOv2 based image encoder.
    
    Uses pretrained DINOv2 model to extract image features.
    Returns patch tokens which can be used for cross-attention conditioning.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        resolution: int = 518,
        output_type: str = "patch_tokens",  # "patch_tokens", "cls_token", or "x_prenorm"
        freeze: bool = True,
    ):
        # Get output dim based on model
        output_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        output_dim = output_dims.get(model_name, 1024)
        
        super().__init__(output_dim)
        
        self.model_name = model_name
        self.resolution = resolution
        self.output_type = output_type
        self.freeze = freeze
        
        # Load pretrained model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Preprocess image(s) for DINOv2.
        
        Args:
            images: Single image or list of images
        
        Returns:
            Preprocessed tensor [B, 3, H, W]
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        tensors = []
        for img in images:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                # Handle RGBA by compositing on white background
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            tensor = self.transform(img)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images.
        
        Args:
            x: Image tensor [B, 3, H, W]
        
        Returns:
            Features tensor:
            - "patch_tokens": [B, N, C] where N = (H/14)*(W/14)
            - "cls_token": [B, C]
            - "x_prenorm": [B, N+1, C] full features before final norm
        """
        x = x.to(self.device)
        
        if self.output_type == "cls_token":
            return self.model(x)
        
        # Get intermediate features
        features = self.model.forward_features(x)
        
        if self.output_type == "x_prenorm":
            # Get features before final layer norm
            x_prenorm = features['x_prenorm']
            # Apply layer norm for stability
            x_prenorm = F.layer_norm(x_prenorm, x_prenorm.shape[-1:])
            return x_prenorm
        
        else:  # patch_tokens
            patch_tokens = features['x_norm_patchtokens']
            return patch_tokens
    
    def encode_images(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Convenience method to preprocess and encode images.
        
        Args:
            images: PIL image(s)
        
        Returns:
            Encoded features
        """
        x = self.preprocess(images).to(self.device)
        return self.forward(x)


class CLIPEncoder(ImageEncoder):
    """
    CLIP based image encoder.
    
    Alternative to DINOv2 for image conditioning.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        resolution: int = 224,
        freeze: bool = True,
    ):
        super().__init__(768)  # CLIP ViT-L has 768 dim
        
        try:
            import clip
        except ImportError:
            raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")
        
        self.model, self.preprocess_fn = clip.load(model_name)
        self.resolution = resolution
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def preprocess(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess images for CLIP."""
        if isinstance(images, Image.Image):
            images = [images]
        
        tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor = self.preprocess_fn(img)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images with CLIP."""
        x = x.to(self.device)
        return self.model.encode_image(x)


class MultiViewEncoder(ImageEncoder):
    """
    Encoder for multiple views of an object.
    
    Aggregates features from multiple views.
    """
    
    def __init__(
        self,
        base_encoder: ImageEncoder,
        aggregation: str = "mean",  # "mean", "concat", "attention"
    ):
        super().__init__(base_encoder.output_dim)
        self.base_encoder = base_encoder
        self.aggregation = aggregation
        
        if aggregation == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=base_encoder.output_dim,
                num_heads=8,
                batch_first=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple views.
        
        Args:
            x: [B, V, 3, H, W] where V is number of views
        
        Returns:
            Aggregated features
        """
        B, V = x.shape[:2]
        
        # Encode each view
        x = x.reshape(B * V, *x.shape[2:])
        features = self.base_encoder(x)  # [B*V, N, C] or [B*V, C]
        
        if features.dim() == 2:
            features = features.reshape(B, V, -1)
        else:
            N, C = features.shape[1:]
            features = features.reshape(B, V, N, C)
        
        # Aggregate
        if self.aggregation == "mean":
            return features.mean(dim=1)
        elif self.aggregation == "concat":
            if features.dim() == 3:
                return features.reshape(B, -1)
            else:
                return features.reshape(B, V * N, C)
        elif self.aggregation == "attention":
            if features.dim() == 3:
                features = features.unsqueeze(2)  # [B, V, 1, C]
            # Self-attention across views
            B, V, N, C = features.shape
            features = features.reshape(B, V * N, C)
            attn_out, _ = self.attention(features, features, features)
            return attn_out.reshape(B, V, N, C).mean(dim=1)
        
        return features

