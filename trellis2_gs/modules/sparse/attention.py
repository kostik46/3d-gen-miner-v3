"""
Sparse Multi-Head Attention.
"""

from typing import Optional, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tensor import SparseTensor


class SparseMultiHeadAttention(nn.Module):
    """
    Multi-head attention for sparse tensors.
    
    Supports:
    - Full attention (all tokens attend to all)
    - Windowed attention (local attention within windows)
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.use_rope = use_rope
        
        self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)
        
        if qk_rms_norm:
            from ..norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        
        if use_rope:
            self._init_rope()
    
    def _init_rope(self):
        """Initialize rotary position embeddings."""
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _apply_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        # coords: [N, 3] (x, y, z)
        # x: [N, H, D]
        
        seq_len = x.shape[0]
        dim = self.head_dim
        
        # Compute position encodings
        pos = coords.float()  # [N, 3]
        freqs = torch.einsum("nd,d->nd", pos[:, 0:1], self.inv_freq)  # Use x coord
        
        # Create rotation matrix
        cos = freqs.cos()  # [N, D/2]
        sin = freqs.sin()  # [N, D/2]
        
        # Split x into pairs
        x1 = x[..., ::2]  # [N, H, D/2]
        x2 = x[..., 1::2]  # [N, H, D/2]
        
        # Apply rotation
        cos = cos.unsqueeze(1)  # [N, 1, D/2]
        sin = sin.unsqueeze(1)  # [N, 1, D/2]
        
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rot
    
    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Full attention within each batch.
        
        Args:
            q, k, v: [N, H, D]
            batch_indices: [N]
            batch_size: Number of batches
        """
        outputs = []
        
        for b in range(batch_size):
            mask = batch_indices == b
            q_b = q[mask]  # [N_b, H, D]
            k_b = k[mask]
            v_b = v[mask]
            
            # Compute attention
            attn = torch.einsum("nhd,mhd->hnm", q_b, k_b) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum("hnm,mhd->nhd", attn, v_b)
            
            outputs.append((mask, out))
        
        # Scatter back
        result = torch.zeros_like(q)
        for mask, out in outputs:
            result[mask] = out
        
        return result
    
    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Windowed local attention.
        """
        window_size = self.window_size
        
        # Compute window indices
        window_coords = coords[:, 1:] // window_size
        window_batch = torch.cat([
            batch_indices.unsqueeze(-1),
            window_coords
        ], dim=-1)
        
        # Find unique windows
        unique_windows, inverse = torch.unique(window_batch, dim=0, return_inverse=True)
        
        outputs = torch.zeros_like(q)
        
        for w in range(unique_windows.shape[0]):
            mask = inverse == w
            q_w = q[mask]
            k_w = k[mask]
            v_w = v[mask]
            
            # Attention within window
            attn = torch.einsum("nhd,mhd->hnm", q_w, k_w) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum("hnm,mhd->nhd", attn, v_w)
            
            outputs[mask] = out
        
        return outputs
    
    def forward(
        self,
        x: SparseTensor,
        context: Optional[torch.Tensor] = None,
    ) -> SparseTensor:
        """
        Forward pass.
        
        Args:
            x: Input sparse tensor
            context: Optional context for cross-attention [B, L, C]
        """
        N = x.feats.shape[0]
        
        if context is None:
            # Self-attention
            qkv = self.to_qkv(x.feats)  # [N, 3C]
            qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [N, H, D]
            
            # Apply RMS norm if enabled
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)
            
            # Apply RoPE if enabled
            if self.use_rope:
                q = self._apply_rope(q, x.coords[:, 1:])
                k = self._apply_rope(k, x.coords[:, 1:])
            
            batch_indices = x.coords[:, 0]
            
            if self.attn_mode == "full":
                out = self._full_attention(q, k, v, batch_indices, x.shape)
            else:
                out = self._windowed_attention(
                    q, k, v, x.coords, batch_indices, x.shape
                )
        else:
            # Cross-attention
            q = self.to_qkv(x.feats)[:, :self.channels]  # Only need Q
            q = q.reshape(N, self.num_heads, self.head_dim)
            
            # Context provides K, V
            B, L, C = context.shape
            kv = self.to_qkv(context)[:, :, self.channels:]
            kv = kv.reshape(B, L, 2, self.num_heads, self.head_dim)
            k, v = kv[:, :, 0], kv[:, :, 1]  # [B, L, H, D]
            
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)
            
            # Cross-attention per batch
            batch_indices = x.coords[:, 0]
            outputs = []
            
            for b in range(B):
                mask = batch_indices == b
                q_b = q[mask]  # [N_b, H, D]
                k_b = k[b]  # [L, H, D]
                v_b = v[b]
                
                attn = torch.einsum("nhd,lhd->hnl", q_b, k_b) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = torch.einsum("hnl,lhd->nhd", attn, v_b)
                
                outputs.append((mask, out))
            
            out = torch.zeros_like(q)
            for mask, o in outputs:
                out[mask] = o
        
        # Reshape and project output
        out = out.reshape(N, self.channels)
        out = self.to_out(out)
        
        return x.replace(out)


class SparseCrossAttention(nn.Module):
    """
    Cross-attention for sparse tensors attending to dense context.
    """
    
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(context_channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)
        
        if qk_rms_norm:
            from ..norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
    
    def forward(
        self,
        x: SparseTensor,
        context: torch.Tensor,
    ) -> SparseTensor:
        """
        Args:
            x: Sparse tensor [N points]
            context: Dense context [B, L, C]
        """
        N = x.feats.shape[0]
        B, L, _ = context.shape
        
        # Query from sparse
        q = self.to_q(x.feats)  # [N, C]
        q = q.reshape(N, self.num_heads, self.head_dim)
        
        # Key, Value from context
        kv = self.to_kv(context)  # [B, L, 2C]
        kv = kv.reshape(B, L, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, L, H, D]
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Cross-attention per batch
        batch_indices = x.coords[:, 0]
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        
        for b in range(B):
            mask = batch_indices == b
            if mask.sum() == 0:
                continue
                
            q_b = q[mask]  # [N_b, H, D]
            k_b = k[b]  # [L, H, D]
            v_b = v[b]
            
            attn = torch.einsum("nhd,lhd->hnl", q_b, k_b) * self.scale
            attn = F.softmax(attn, dim=-1)
            out_b = torch.einsum("hnl,lhd->nhd", attn, v_b)
            
            out[mask] = out_b
        
        out = out.reshape(N, self.channels)
        out = self.to_out(out)
        
        return x.replace(out)

