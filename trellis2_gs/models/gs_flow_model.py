"""
Gaussian Splatting Flow Model.

Generates latent codes that decode to Gaussian parameters.
Uses sparse transformer architecture operating on voxel coordinates.
"""

from typing import Optional, Literal, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.utils import zero_module, convert_module_to_fp16, convert_module_to_fp32, str_to_dtype, manual_cast
from ..modules.norm import LayerNorm32
from ..modules.sparse import SparseTensor, SparseLinear
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AbsolutePositionEmbedder(nn.Module):
    """
    Absolute position embeddings for 3D coordinates.
    """
    
    def __init__(self, channels: int, max_position: int = 1024):
        super().__init__()
        self.channels = channels
        
        # Create learnable embeddings for each dimension
        self.embed_x = nn.Embedding(max_position, channels // 3)
        self.embed_y = nn.Embedding(max_position, channels // 3)
        self.embed_z = nn.Embedding(max_position, channels - 2 * (channels // 3))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [N, 3] integer coordinates
        
        Returns:
            Position embeddings [N, channels]
        """
        x = coords[:, 0].long()
        y = coords[:, 1].long()
        z = coords[:, 2].long()
        
        emb_x = self.embed_x(x)
        emb_y = self.embed_y(y)
        emb_z = self.embed_z(z)
        
        return torch.cat([emb_x, emb_y, emb_z], dim=-1)


class GSFlowModel(nn.Module):
    """
    Flow model for generating Gaussian Splatting latent codes.
    
    Takes:
    - Sparse voxel coordinates (from Sparse Structure Flow)
    - Image conditioning
    - Timestep
    
    Outputs:
    - Latent codes that decode to Gaussian parameters
    """
    
    def __init__(
        self,
        resolution: int = 64,
        in_channels: int = 8,
        out_channels: int = 8,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        pe_mode: Literal["ape", "rope"] = "ape",
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        self.dtype = str_to_dtype(dtype)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # Adaptive layer norm modulation
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        
        # Position embedding
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        
        # Input projection
        self.input_layer = SparseLinear(in_channels, model_channels)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode='full',
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.out_layer = SparseLinear(model_channels, out_channels)
        
        self.initialize_weights()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layer
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    
    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        **kwargs,
    ) -> SparseTensor:
        """
        Forward pass.
        
        Args:
            x: Noisy sparse latent
            t: Timestep [B]
            cond: Image conditioning [B, L, cond_C]
        
        Returns:
            Predicted velocity (sparse tensor)
        """
        # Input projection
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        
        # Timestep embedding
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)
        
        # Position embedding
        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, cond)
        
        # Output
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        
        return h


class ConditionalGSFlowModel(GSFlowModel):
    """
    GS Flow Model with additional shape conditioning.
    
    Can be conditioned on shape latent from a separate model.
    """
    
    def __init__(
        self,
        shape_cond_channels: int = 8,
        **kwargs
    ):
        # Adjust in_channels to account for shape conditioning
        in_channels = kwargs.get('in_channels', 8)
        kwargs['in_channels'] = in_channels + shape_cond_channels
        
        super().__init__(**kwargs)
        
        self.shape_cond_channels = shape_cond_channels
    
    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        shape_cond: Optional[SparseTensor] = None,
        **kwargs,
    ) -> SparseTensor:
        """
        Forward pass with optional shape conditioning.
        
        Args:
            x: Noisy sparse latent
            t: Timestep [B]
            cond: Image conditioning [B, L, cond_C]
            shape_cond: Shape latent (optional)
        
        Returns:
            Predicted velocity
        """
        if shape_cond is not None:
            # Concatenate shape conditioning
            x = x.replace(torch.cat([x.feats, shape_cond.feats], dim=-1))
        
        return super().forward(x, t, cond, **kwargs)

