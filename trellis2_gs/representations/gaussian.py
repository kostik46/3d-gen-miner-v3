"""
3D Gaussian Splatting representation.
"""

from typing import Optional, List, Union
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import utils3d


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse of sigmoid function."""
    return torch.log(x / (1 - x))


def build_rotation_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix from quaternion.
    
    Args:
        r: Quaternion [N, 4] in (w, x, y, z) format
    
    Returns:
        Rotation matrix [N, 3, 3]
    """
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    
    R = torch.zeros((q.shape[0], 3, 3), device=r.device)
    
    r0, r1, r2, r3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R[:, 0, 0] = 1 - 2 * (r2 * r2 + r3 * r3)
    R[:, 0, 1] = 2 * (r1 * r2 - r3 * r0)
    R[:, 0, 2] = 2 * (r1 * r3 + r2 * r0)
    R[:, 1, 0] = 2 * (r1 * r2 + r3 * r0)
    R[:, 1, 1] = 1 - 2 * (r1 * r1 + r3 * r3)
    R[:, 1, 2] = 2 * (r2 * r3 - r1 * r0)
    R[:, 2, 0] = 2 * (r1 * r3 - r2 * r0)
    R[:, 2, 1] = 2 * (r2 * r3 + r1 * r0)
    R[:, 2, 2] = 1 - 2 * (r1 * r1 + r2 * r2)
    
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Build covariance matrix from scaling and rotation.
    
    Args:
        s: Scaling [N, 3]
        r: Quaternion [N, 4]
    
    Returns:
        L matrix [N, 3, 3] such that covariance = L @ L.T
    """
    L = torch.zeros((s.shape[0], 3, 3), device=s.device)
    R = build_rotation_matrix(r)
    
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    
    L = R @ L
    return L


class Gaussian:
    """
    3D Gaussian Splatting model.
    
    Attributes:
        _xyz: Raw position offsets [N, 3]
        _features_dc: DC spherical harmonic coefficients [N, 1, 3]
        _features_rest: Higher-order SH coefficients [N, (deg+1)^2-1, 3] (optional)
        _scaling: Raw scaling values [N, 3]
        _rotation: Raw quaternion values [N, 4]
        _opacity: Raw opacity values [N, 1]
    """
    
    def __init__(
        self,
        aabb: List[float],
        sh_degree: int = 0,
        mininum_kernel_size: float = 0.0,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
        device: str = 'cuda',
    ):
        """
        Initialize Gaussian model.
        
        Args:
            aabb: Axis-aligned bounding box [min_x, min_y, min_z, size_x, size_y, size_z]
            sh_degree: Spherical harmonic degree (0 = only DC)
            mininum_kernel_size: Minimum kernel size for stability
            scaling_bias: Bias for scaling initialization
            opacity_bias: Bias for opacity initialization
            scaling_activation: Activation for scaling ("exp" or "softplus")
            device: Device to store tensors on
        """
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        
        # Setup activation functions
        self._setup_activations()
        
        # Initialize empty attributes
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
    
    def _setup_activations(self):
        """Setup activation functions based on configuration."""
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))
        
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        # Compute biases
        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias, device=self.device)
        )
        self.rots_bias = torch.zeros(4, device=self.device)
        self.rots_bias[0] = 1.0  # Identity quaternion
        self.opacity_bias_value = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias, device=self.device)
        )
    
    def to(self, device: str) -> "Gaussian":
        """Move to device."""
        self.device = device
        self.aabb = self.aabb.to(device)
        self.scale_bias = self.scale_bias.to(device)
        self.rots_bias = self.rots_bias.to(device)
        self.opacity_bias_value = self.opacity_bias_value.to(device)
        
        if self._xyz is not None:
            self._xyz = self._xyz.to(device)
        if self._features_dc is not None:
            self._features_dc = self._features_dc.to(device)
        if self._features_rest is not None:
            self._features_rest = self._features_rest.to(device)
        if self._scaling is not None:
            self._scaling = self._scaling.to(device)
        if self._rotation is not None:
            self._rotation = self._rotation.to(device)
        if self._opacity is not None:
            self._opacity = self._opacity.to(device)
        
        return self
    
    def cuda(self) -> "Gaussian":
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "Gaussian":
        """Move to CPU."""
        return self.to('cpu')
    
    @property
    def num_gaussians(self) -> int:
        """Number of gaussians."""
        if self._xyz is None:
            return 0
        return self._xyz.shape[0]
    
    @property
    def get_xyz(self) -> torch.Tensor:
        """Get world-space positions."""
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self) -> torch.Tensor:
        """Get all SH features."""
        if self._features_rest is not None:
            return torch.cat([self._features_dc, self._features_rest], dim=1)
        return self._features_dc
    
    @property
    def get_scaling(self) -> torch.Tensor:
        """Get activated scaling."""
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self) -> torch.Tensor:
        """Get normalized rotation quaternion."""
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_opacity(self) -> torch.Tensor:
        """Get activated opacity."""
        return self.opacity_activation(self._opacity + self.opacity_bias_value)
    
    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        Get 3D covariance matrices.
        
        Args:
            scaling_modifier: Scale factor for covariance
        
        Returns:
            Covariance matrices [N, 6] (symmetric, upper triangular)
        """
        L = build_scaling_rotation(
            scaling_modifier * self.get_scaling,
            self._rotation + self.rots_bias[None, :]
        )
        cov = L @ L.transpose(1, 2)
        
        # Return upper triangular
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]
        ], dim=1)
    
    # Setters from world-space values
    def from_xyz(self, xyz: torch.Tensor):
        """Set positions from world-space coordinates."""
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
    
    def from_features(self, features: torch.Tensor):
        """Set DC features."""
        self._features_dc = features
    
    def from_scaling(self, scales: torch.Tensor):
        """Set scaling from world-space scales."""
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
    
    def from_rotation(self, rots: torch.Tensor):
        """Set rotation from normalized quaternions."""
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_opacity(self, opacities: torch.Tensor):
        """Set opacity from activated values."""
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias_value
    
    def construct_list_of_attributes(self) -> List[str]:
        """Get list of PLY attribute names."""
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        # DC features
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append(f'f_dc_{i}')
        
        # Rest features
        if self._features_rest is not None:
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                attrs.append(f'f_rest_{i}')
        
        attrs.append('opacity')
        
        for i in range(self._scaling.shape[1]):
            attrs.append(f'scale_{i}')
        
        for i in range(self._rotation.shape[1]):
            attrs.append(f'rot_{i}')
        
        return attrs
    
    def save_ply(
        self,
        path: str,
        transform: Optional[List[List[float]]] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    ):
        """
        Save to PLY file.
        
        Args:
            path: Output path
            transform: Optional 3x3 rotation matrix for coordinate transform
        """
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)
        
        dtype_full = [(attr, 'f4') for attr in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        if self._features_rest is not None:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
        else:
            attributes = np.concatenate([xyz, normals, f_dc, opacities, scale, rotation], axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(
        self,
        path: str,
        transform: Optional[List[List[float]]] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    ):
        """
        Load from PLY file.
        
        Args:
            path: Input path
            transform: Optional 3x3 rotation matrix for coordinate transform
        """
        plydata = PlyData.read(path)
        
        xyz = np.stack([
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ], axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(sorted(scale_names)):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(sorted(rot_names)):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rots = utils3d.numpy.quaternion_to_matrix(rots)
            rots = np.matmul(rots, transform)
            rots = utils3d.numpy.matrix_to_quaternion(rots)
        
        # Convert to tensors
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # Set internal attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias_value
        self._scaling = self.inverse_scaling_activation(
            torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        ) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]
    
    def clone(self) -> "Gaussian":
        """Create a deep copy."""
        g = Gaussian(
            aabb=self.aabb.tolist(),
            sh_degree=self.sh_degree,
            mininum_kernel_size=self.mininum_kernel_size,
            scaling_bias=self.scaling_bias,
            opacity_bias=self.opacity_bias,
            scaling_activation=self.scaling_activation_type,
            device=self.device,
        )
        
        if self._xyz is not None:
            g._xyz = self._xyz.clone()
        if self._features_dc is not None:
            g._features_dc = self._features_dc.clone()
        if self._features_rest is not None:
            g._features_rest = self._features_rest.clone()
        if self._scaling is not None:
            g._scaling = self._scaling.clone()
        if self._rotation is not None:
            g._rotation = self._rotation.clone()
        if self._opacity is not None:
            g._opacity = self._opacity.clone()
        
        return g
    
    def __repr__(self) -> str:
        return f"Gaussian(num_gaussians={self.num_gaussians}, sh_degree={self.sh_degree})"

