"""
Gaussian Splatting Decoder.

Decodes latent codes to Gaussian parameters (position, color, scale, rotation, opacity).
"""

from typing import Optional, List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.utils import zero_module
from ..modules.sparse import SparseTensor, SparseLinear
from ..modules.sparse.transformer import SparseTransformerBlock
from ..modules.norm import LayerNorm32
from ..representations.gaussian import Gaussian
from ..utils.random_utils import hammersley_sequence


class AbsolutePositionEmbedder(nn.Module):
    """Absolute position embeddings for 3D coordinates."""
    
    def __init__(self, channels: int, max_position: int = 1024):
        super().__init__()
        self.channels = channels
        self.embed_x = nn.Embedding(max_position, channels // 3)
        self.embed_y = nn.Embedding(max_position, channels // 3)
        self.embed_z = nn.Embedding(max_position, channels - 2 * (channels // 3))
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0].long()
        y = coords[:, 1].long()
        z = coords[:, 2].long()
        return torch.cat([
            self.embed_x(x),
            self.embed_y(y),
            self.embed_z(z)
        ], dim=-1)


class GSDecoder(nn.Module):
    """
    Decodes sparse latent to Gaussian parameters.
    
    For each voxel in the sparse tensor, produces multiple Gaussians
    with position offsets, colors, scales, rotations, and opacities.
    """
    
    def __init__(
        self,
        resolution: int = 64,
        model_channels: int = 512,
        latent_channels: int = 8,
        num_blocks: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed", "swin"] = "full",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        # Gaussian configuration
        num_gaussians: int = 4,
        voxel_size: float = 0.015625,  # 1/64
        perturb_offset: bool = True,
        # Representation config
        sh_degree: int = 0,
        min_kernel_size: float = 0.0001,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
    ):
        super().__init__()
        
        self.resolution = resolution
        self.model_channels = model_channels
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Gaussian config
        self.num_gaussians = num_gaussians
        self.voxel_size = voxel_size
        self.perturb_offset = perturb_offset
        
        # Representation config
        self.rep_config = {
            'num_gaussians': num_gaussians,
            'voxel_size': voxel_size,
            'perturb_offset': perturb_offset,
            'sh_degree': sh_degree,
            '3d_filter_kernel_size': min_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
            # Learning rates for different parameters
            'lr': {
                '_xyz': 1.0,
                '_features_dc': 1.0,
                '_scaling': 1.0,
                '_rotation': 1.0,
                '_opacity': 1.0,
            }
        }
        
        # Calculate output layout
        self._calc_layout()
        
        # Position embedding
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.pe_mode = pe_mode
        
        # Input layer
        self.input_layer = SparseLinear(latent_channels, model_channels)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm,
            )
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.out_layer = SparseLinear(model_channels, self.out_channels)
        
        # Build perturbation offsets
        self._build_perturbation()
        
        self.initialize_weights()
        
        if use_fp16:
            self.convert_to_fp16()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _calc_layout(self):
        """Calculate output channel layout for Gaussian parameters."""
        ng = self.num_gaussians
        
        self.layout = {
            '_xyz': {'shape': (ng, 3), 'size': ng * 3},
            '_features_dc': {'shape': (ng, 1, 3), 'size': ng * 3},
            '_scaling': {'shape': (ng, 3), 'size': ng * 3},
            '_rotation': {'shape': (ng, 4), 'size': ng * 4},
            '_opacity': {'shape': (ng, 1), 'size': ng},
        }
        
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        
        self.out_channels = start
    
    def _build_perturbation(self):
        """Build deterministic perturbation offsets using Hammersley sequence."""
        perturbation = [
            hammersley_sequence(3, i, self.num_gaussians)
            for i in range(self.num_gaussians)
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.voxel_size
        perturbation = torch.atanh(perturbation.clamp(-0.99, 0.99))
        self.register_buffer('offset_perturbation', perturbation)
    
    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out output layer
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    
    def convert_to_fp16(self):
        self.blocks.apply(lambda m: m.half() if hasattr(m, 'half') else None)
    
    def convert_to_fp32(self):
        self.blocks.apply(lambda m: m.float() if hasattr(m, 'float') else None)
    
    def to_representation(self, x: SparseTensor) -> List[Gaussian]:
        """
        Convert decoder output to list of Gaussian objects.
        
        Args:
            x: Decoder output sparse tensor [N x out_channels]
        
        Returns:
            List of Gaussian objects (one per batch)
        """
        ret = []
        
        for i in range(x.shape):
            # Create Gaussian with config
            gaussian = Gaussian(
                sh_degree=self.rep_config.get('sh_degree', 0),
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                min_kernel_size=self.rep_config['3d_filter_kernel_size'],
                scaling_bias=self.rep_config['scaling_bias'],
                opacity_bias=self.rep_config['opacity_bias'],
                scaling_activation=self.rep_config['scaling_activation'],
                device=x.device,
            )
            
            # Get indices for this batch
            batch_mask = x.layout[i]
            batch_coords = x.coords[batch_mask][:, 1:]  # [N_i, 3]
            batch_feats = x.feats[batch_mask]  # [N_i, out_channels]
            
            # Base voxel positions (normalized to [0, 1])
            xyz_base = (batch_coords.float() + 0.5) / self.resolution
            
            # Process each parameter
            for k, v in self.layout.items():
                feat_slice = batch_feats[:, v['range'][0]:v['range'][1]]
                feat_reshaped = feat_slice.reshape(-1, *v['shape'])
                
                if k == '_xyz':
                    # Position offsets
                    offset = feat_reshaped * self.rep_config['lr'][k]
                    
                    if self.perturb_offset:
                        offset = offset + self.offset_perturbation.to(offset.device)
                    
                    # Tanh activation and scale by voxel size
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.voxel_size
                    
                    # Add to base position
                    _xyz = xyz_base.unsqueeze(1) + offset  # [N_i, ng, 3]
                    gaussian._xyz = _xyz.flatten(0, 1)  # [N_i * ng, 3]
                    
                else:
                    # Other parameters
                    feats = feat_reshaped.flatten(0, 1)  # [N_i * ng, ...]
                    feats = feats * self.rep_config['lr'][k]
                    setattr(gaussian, k, feats)
            
            ret.append(gaussian)
        
        return ret
    
    def forward(self, x: SparseTensor) -> List[Gaussian]:
        """
        Decode sparse latent to Gaussians.
        
        Args:
            x: Sparse latent tensor
        
        Returns:
            List of Gaussian objects
        """
        # Input projection
        h = self.input_layer(x)
        
        # Position embedding
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        
        h = h.type(self.dtype)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Output
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        
        return self.to_representation(h)


class LightweightGSDecoder(nn.Module):
    """
    Lightweight GS decoder using only MLPs.
    
    Faster than transformer-based decoder but may have lower quality.
    """
    
    def __init__(
        self,
        resolution: int = 64,
        latent_channels: int = 8,
        hidden_channels: int = 256,
        num_layers: int = 4,
        num_gaussians: int = 4,
        voxel_size: float = 0.015625,
    ):
        super().__init__()
        
        self.resolution = resolution
        self.num_gaussians = num_gaussians
        self.voxel_size = voxel_size
        
        # Output: xyz(3) + rgb(3) + scale(3) + rot(4) + opacity(1) = 14 per gaussian
        out_per_gaussian = 14
        out_channels = num_gaussians * out_per_gaussian
        
        layers = [nn.Linear(latent_channels + 3, hidden_channels), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.SiLU()])
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.mlp = nn.Sequential(*layers)
        
        # Build perturbation
        perturbation = [
            hammersley_sequence(3, i, num_gaussians)
            for i in range(num_gaussians)
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        self.register_buffer('offset_perturbation', perturbation)
    
    def forward(self, x: SparseTensor) -> List[Gaussian]:
        """Decode to Gaussians."""
        ret = []
        
        for i in range(x.shape):
            batch_mask = x.layout[i]
            batch_coords = x.coords[batch_mask][:, 1:]  # [N, 3]
            batch_feats = x.feats[batch_mask]  # [N, C]
            
            # Normalize coordinates and concatenate
            coords_norm = batch_coords.float() / self.resolution
            input_feat = torch.cat([batch_feats, coords_norm], dim=-1)
            
            # MLP forward
            out = self.mlp(input_feat)  # [N, ng * 14]
            out = out.reshape(-1, self.num_gaussians, 14)  # [N, ng, 14]
            
            # Parse outputs
            xyz_offset = torch.tanh(out[..., :3]) * self.voxel_size
            rgb = torch.sigmoid(out[..., 3:6])
            scale = out[..., 6:9]
            rotation = out[..., 9:13]
            opacity = out[..., 13:14]
            
            # Add perturbation to xyz
            xyz_base = (batch_coords.float() + 0.5) / self.resolution
            xyz = xyz_base.unsqueeze(1) + xyz_offset + self.offset_perturbation * self.voxel_size
            
            # Create Gaussian
            gaussian = Gaussian(device=x.device)
            gaussian._xyz = xyz.flatten(0, 1)
            gaussian._features_dc = rgb.flatten(0, 1).unsqueeze(1)
            gaussian._scaling = scale.flatten(0, 1)
            gaussian._rotation = rotation.flatten(0, 1)
            gaussian._opacity = opacity.flatten(0, 1)
            
            ret.append(gaussian)
        
        return ret

