"""
Sparse Transformer blocks.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tensor import SparseTensor
from .attention import SparseMultiHeadAttention, SparseCrossAttention
from ..norm import LayerNorm32


class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """
    
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Basic sparse transformer block.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(channels, mlp_ratio)
    
    def _forward(self, x: SparseTensor) -> SparseTensor:
        # Self-attention
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h)
        x = x + h
        
        # MLP
        h = x.replace(self.norm2(x.feats))
        h = x.replace(self.mlp(h.feats))
        x = x + h
        
        return x
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse transformer block with adaptive modulation (AdaLN).
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(channels, mlp_ratio)
        
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
    
    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
    ) -> SparseTensor:
        """
        Args:
            x: Sparse tensor
            mod: Modulation tensor [B, 6C] or pre-split [B, 6, C]
        """
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        
        if mod.dim() == 2:
            mod = mod.reshape(mod.shape[0], 6, -1)
        
        # Get per-point modulation based on batch index
        batch_idx = x.coords[:, 0].long()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod[batch_idx, 0], mod[batch_idx, 1], mod[batch_idx, 2], \
            mod[batch_idx, 3], mod[batch_idx, 4], mod[batch_idx, 5]
        
        # Self-attention with modulation
        h = x.replace(self.norm1(x.feats))
        h = h.replace(h.feats * (1 + scale_msa) + shift_msa)
        h = self.attn(h)
        h = h.replace(h.feats * gate_msa)
        x = x + h
        
        # MLP with modulation
        h = x.replace(self.norm2(x.feats))
        h = h.replace(h.feats * (1 + scale_mlp) + shift_mlp)
        h = x.replace(self.mlp(h.feats))
        h = h.replace(h.feats * gate_mlp)
        x = x + h
        
        return x
    
    def forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
    ) -> SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, use_reentrant=False
            )
        return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse transformer block with cross-attention and adaptive modulation.
    """
    
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        
        # Self-attention
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Cross-attention
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = SparseCrossAttention(
            channels,
            context_channels,
            num_heads=num_heads,
            qk_rms_norm=qk_rms_norm_cross,
        )
        
        # MLP
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(channels, mlp_ratio)
        
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
    
    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: torch.Tensor,
    ) -> SparseTensor:
        """
        Args:
            x: Sparse tensor
            mod: Modulation tensor [B, C] for timestep/class embedding
            context: Cross-attention context [B, L, C]
        """
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        
        if mod.dim() == 2:
            mod = mod.reshape(mod.shape[0], 6, -1)
        
        batch_idx = x.coords[:, 0].long()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod[batch_idx, 0], mod[batch_idx, 1], mod[batch_idx, 2], \
            mod[batch_idx, 3], mod[batch_idx, 4], mod[batch_idx, 5]
        
        # Self-attention
        h = x.replace(self.norm1(x.feats))
        h = h.replace(h.feats * (1 + scale_msa) + shift_msa)
        h = self.self_attn(h)
        h = h.replace(h.feats * gate_msa)
        x = x + h
        
        # Cross-attention (no modulation)
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        
        # MLP
        h = x.replace(self.norm3(x.feats))
        h = h.replace(h.feats * (1 + scale_mlp) + shift_mlp)
        h = x.replace(self.mlp(h.feats))
        h = h.replace(h.feats * gate_mlp)
        x = x + h
        
        return x
    
    def forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: torch.Tensor,
    ) -> SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        return self._forward(x, mod, context)

