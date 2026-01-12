"""
Sparse Tensor implementation for 3D neural networks.
"""

from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn


class SparseTensor:
    """
    Sparse tensor representation for 3D data.
    
    Attributes:
        feats: Feature tensor of shape [N, C] where N is number of active points
        coords: Coordinate tensor of shape [N, 4] where columns are [batch_idx, x, y, z]
        shape: Batch size
        spatial_shape: Spatial dimensions [D, H, W]
    """
    
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: Optional[int] = None,
        spatial_shape: Optional[tuple] = None,
        layout: Optional[List[torch.Tensor]] = None,
        spatial_cache: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SparseTensor.
        
        Args:
            feats: Feature tensor [N, C]
            coords: Coordinate tensor [N, 4] with [batch_idx, x, y, z]
            shape: Batch size
            spatial_shape: Spatial dimensions
            layout: Per-batch indices
            spatial_cache: Cached spatial operations
        """
        self.feats = feats
        self.coords = coords
        
        if shape is None:
            if coords.shape[0] > 0:
                shape = coords[:, 0].max().item() + 1
            else:
                shape = 1
        self._shape = int(shape)
        
        self._spatial_shape = spatial_shape
        self._layout = layout
        self._spatial_cache = spatial_cache or {}
        
    @property
    def shape(self) -> int:
        """Batch size."""
        return self._shape
    
    @property
    def spatial_shape(self) -> tuple:
        """Spatial dimensions [D, H, W]."""
        if self._spatial_shape is None:
            if self.coords.shape[0] > 0:
                max_coords = self.coords[:, 1:].max(dim=0).values
                self._spatial_shape = tuple(max_coords.tolist())
            else:
                self._spatial_shape = (0, 0, 0)
        return self._spatial_shape
    
    @property
    def layout(self) -> List[torch.Tensor]:
        """Per-batch indices."""
        if self._layout is None:
            self._layout = []
            for i in range(self._shape):
                self._layout.append(self.coords[:, 0] == i)
        return self._layout
    
    @property
    def device(self) -> torch.device:
        """Device of the tensor."""
        return self.feats.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of features."""
        return self.feats.dtype
    
    def to(self, device: Union[str, torch.device]) -> "SparseTensor":
        """Move to device."""
        return SparseTensor(
            feats=self.feats.to(device),
            coords=self.coords.to(device),
            shape=self._shape,
            spatial_shape=self._spatial_shape,
            layout=None,  # Recompute on new device
            spatial_cache=None,
        )
    
    def cuda(self) -> "SparseTensor":
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "SparseTensor":
        """Move to CPU."""
        return self.to('cpu')
    
    def type(self, dtype: torch.dtype) -> "SparseTensor":
        """Cast features to dtype."""
        return self.replace(self.feats.type(dtype))
    
    def float(self) -> "SparseTensor":
        """Cast to float32."""
        return self.type(torch.float32)
    
    def half(self) -> "SparseTensor":
        """Cast to float16."""
        return self.type(torch.float16)
    
    def replace(self, feats: torch.Tensor) -> "SparseTensor":
        """
        Create new SparseTensor with replaced features but same coords.
        """
        return SparseTensor(
            feats=feats,
            coords=self.coords,
            shape=self._shape,
            spatial_shape=self._spatial_shape,
            layout=self._layout,
            spatial_cache=self._spatial_cache,
        )
    
    def get_spatial_cache(self, key: str) -> Optional[Any]:
        """Get cached spatial operation result."""
        return self._spatial_cache.get(key)
    
    def set_spatial_cache(self, key: str, value: Any) -> None:
        """Set cached spatial operation result."""
        self._spatial_cache[key] = value
        
    def __len__(self) -> int:
        """Number of active points."""
        return self.coords.shape[0]
    
    def __repr__(self) -> str:
        return (
            f"SparseTensor(shape={self._shape}, "
            f"spatial_shape={self.spatial_shape}, "
            f"num_points={len(self)}, "
            f"feat_dim={self.feats.shape[-1]})"
        )
    
    def __add__(self, other: "SparseTensor") -> "SparseTensor":
        """Element-wise addition."""
        assert torch.equal(self.coords, other.coords), "Coords must match for addition"
        return self.replace(self.feats + other.feats)
    
    def __mul__(self, other: Union[float, torch.Tensor]) -> "SparseTensor":
        """Scalar or element-wise multiplication."""
        if isinstance(other, SparseTensor):
            return self.replace(self.feats * other.feats)
        return self.replace(self.feats * other)
    
    def __rmul__(self, other: Union[float, torch.Tensor]) -> "SparseTensor":
        """Right multiplication."""
        return self.__mul__(other)
    
    def __getitem__(self, idx: int) -> "SparseTensor":
        """Get single batch element."""
        mask = self.coords[:, 0] == idx
        return SparseTensor(
            feats=self.feats[mask],
            coords=self.coords[mask].clone(),
            shape=1,
        )


def sparse_cat(tensors: List[SparseTensor], dim: int = -1) -> SparseTensor:
    """
    Concatenate sparse tensors along feature dimension.
    """
    assert all(torch.equal(t.coords, tensors[0].coords) for t in tensors), \
        "All tensors must have same coords"
    feats = torch.cat([t.feats for t in tensors], dim=dim)
    return tensors[0].replace(feats)

