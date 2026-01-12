"""
Hybrid Trellis Pipeline.

Uses:
- Sparse Structure from TRELLIS.2-4B (better geometry)
- SLat Flow + GS Decoder from TRELLIS-image-large (native Gaussian Splatting)

All models are loaded locally - no external dependencies.
"""

from typing import *
from contextlib import contextmanager
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from . import samplers
from . import samplers_t2
from ..modules import sparse as sp
from ..representations import Gaussian
from .. import models as trellis1_models
from .. import models_t2 as trellis2_models
from ..modules_t2 import image_feature_extractor


class HybridTrellisGSPipeline:
    """
    Hybrid Pipeline for generating Gaussian Splatting from images.
    
    Uses TRELLIS.2 for sparse structure (better geometry)
    and TRELLIS1 for SLat flow + GS decoder (native Gaussian output).
    """
    
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler = None,
        slat_sampler = None,
        sparse_structure_sampler_params: dict = None,
        slat_sampler_params: dict = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        self.slat_sampler_params = slat_sampler_params or {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(
        trellis2_path: str = "microsoft/TRELLIS.2-4B",
        trellis1_path: str = "microsoft/TRELLIS-image-large",
        **kwargs,  # Accept extra args for backwards compatibility
    ) -> "HybridTrellisGSPipeline":
        """
        Load hybrid pretrained model.

        Args:
            trellis2_path: Path to TRELLIS.2 model (for sparse structure)
            trellis1_path: Path to TRELLIS1 model (for SLat + GS decoder)
        """
        from huggingface_hub import hf_hub_download
        
        pipeline = HybridTrellisGSPipeline()
        _models = {}
        
        # =====================================================================
        # Load TRELLIS.2 models (sparse structure) - using local models_t2
        # =====================================================================
        print(f"Loading sparse structure from {trellis2_path}...")
        
        # Load TRELLIS.2 pipeline config
        if os.path.exists(f"{trellis2_path}/pipeline.json"):
            t2_config_file = f"{trellis2_path}/pipeline.json"
        else:
            t2_config_file = hf_hub_download(trellis2_path, "pipeline.json")
        
        with open(t2_config_file, 'r') as f:
            t2_args = json.load(f)['args']
        
        # Load all TRELLIS.2 models (sparse structure flow + decoder)
        for model_key in ['sparse_structure_flow_model', 'sparse_structure_decoder']:
            model_name = t2_args['models'][model_key]
            # Try path/model_name first, then just model_name
            try:
                _models[model_key] = trellis2_models.from_pretrained(
                    f"{trellis2_path}/{model_name}"
                )
                print(f"  ✓ Loaded {model_key} from {trellis2_path}/{model_name}")
            except Exception as e1:
                try:
                    _models[model_key] = trellis2_models.from_pretrained(model_name)
                    print(f"  ✓ Loaded {model_key} from {model_name}")
                except Exception as e2:
                    print(f"  ✗ Failed to load {model_key}: {e1} / {e2}")
                    raise
        
        # =====================================================================
        # Load TRELLIS1 models (SLat flow + GS decoder) - using local models
        # =====================================================================
        print(f"Loading SLat + GS decoder from {trellis1_path}...")
        
        # Load TRELLIS1 pipeline config
        if os.path.exists(f"{trellis1_path}/pipeline.json"):
            t1_config_file = f"{trellis1_path}/pipeline.json"
        else:
            t1_config_file = hf_hub_download(trellis1_path, "pipeline.json")
        
        with open(t1_config_file, 'r') as f:
            t1_args = json.load(f)['args']
        
        # Load TRELLIS1 models (slat flow + GS decoder)
        for model_key in ['slat_flow_model', 'slat_decoder_gs']:
            model_name = t1_args['models'][model_key]
            # Try path/model_name first, then just model_name
            try:
                _models[model_key] = trellis1_models.from_pretrained(
                    f"{trellis1_path}/{model_name}"
                )
                print(f"  ✓ Loaded {model_key} from {trellis1_path}/{model_name}")
            except Exception as e1:
                try:
                    _models[model_key] = trellis1_models.from_pretrained(model_name)
                    print(f"  ✓ Loaded {model_key} from {model_name}")
                except Exception as e2:
                    print(f"  ✗ Failed to load {model_key}: {e1} / {e2}")
                    raise
        
        # =====================================================================
        # Setup pipeline
        # =====================================================================
        pipeline.models = _models
        for model in pipeline.models.values():
            model.eval()
        pipeline._pretrained_args_t2 = t2_args
        pipeline._pretrained_args_t1 = t1_args
        
        # Use TRELLIS.2 sparse structure sampler (from local samplers_t2)
        pipeline.sparse_structure_sampler = getattr(
            samplers_t2, 
            t2_args['sparse_structure_sampler']['name']
        )(**t2_args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = t2_args['sparse_structure_sampler']['params']
        
        # Use TRELLIS1 slat sampler (local)
        pipeline.slat_sampler = getattr(
            samplers, 
            t1_args['slat_sampler']['name']
        )(**t1_args['slat_sampler']['args'])
        pipeline.slat_sampler_params = t1_args['slat_sampler']['params']
        
        # Use TRELLIS1 slat normalization
        pipeline.slat_normalization = t1_args['slat_normalization']
        
        # Use TRELLIS.2 image cond model (DINOv3) for sparse structure
        t2_img_cond_args = t2_args['image_cond_model']
        pipeline.image_cond_model_t2 = getattr(
            image_feature_extractor, 
            t2_img_cond_args['name']
        )(**t2_img_cond_args['args'])
        print(f"  ✓ Loaded image_cond_model_t2 (DINOv3): {t2_img_cond_args['name']}")
        
        # Use TRELLIS1 image cond model (DINOv2) for slat - CRITICAL for colors!
        t1_img_cond_name = t1_args['image_cond_model']
        pipeline.image_cond_model_t1 = torch.hub.load('facebookresearch/dinov2', t1_img_cond_name, pretrained=True)
        pipeline.image_cond_model_t1.eval()
        pipeline.image_cond_model_t1_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"  ✓ Loaded image_cond_model_t1 (DINOv2): {t1_img_cond_name}")

        pipeline._device = torch.device('cpu')
        
        print("=" * 60)
        print("HYBRID PIPELINE LOADED SUCCESSFULLY!")
        print("=" * 60)
        print(f"  Sparse Structure: {trellis2_path} (TRELLIS.2)")
        print(f"  SLat + GS Decoder: {trellis1_path} (TRELLIS1)")
        print("=" * 60)
        
        return pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> "HybridTrellisGSPipeline":
        self._device = device
        for model in self.models.values():
            model.to(device)
        self.image_cond_model_t2.to(device)
        self.image_cond_model_t1.to(device)
        return self

    def cuda(self) -> "HybridTrellisGSPipeline":
        return self.to(torch.device("cuda"))

    def cpu(self) -> "HybridTrellisGSPipeline":
        return self.to(torch.device("cpu"))

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.

        NOTE: This method is deprecated. Use external BEN2 + center_and_crop
        from modules.utils instead. This is kept for backwards compatibility.

        Returns RGB image.
        """
        # Just convert to RGB - preprocessing should be done externally
        if input.mode == 'RGBA':
            # Premultiply alpha
            import numpy as np
            img_array = np.array(input).astype(np.float32) / 255.0
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3:4]
            premultiplied = rgb * alpha
            return Image.fromarray((premultiplied * 255).astype(np.uint8))
        return input.convert('RGB')

    def get_cond_t2(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int = 512) -> dict:
        """Get conditioning using TRELLIS.2 image_cond_model (DINOv3) for sparse structure."""
        self.image_cond_model_t2.image_size = resolution
        cond = self.image_cond_model_t2(image)
        neg_cond = torch.zeros_like(cond)
        return {'cond': cond, 'neg_cond': neg_cond}
    
    @torch.no_grad()
    def get_cond_t1(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """Get conditioning using TRELLIS1 image_cond_model (DINOv2) for slat."""
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
        elif isinstance(image, list):
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        
        image = self.image_cond_model_t1_transform(image).to(self.device)
        features = self.image_cond_model_t1(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        neg_cond = torch.zeros_like(patchtokens)
        return {'cond': patchtokens, 'neg_cond': neg_cond}

    def sample_sparse_structure(
        self, 
        cond: dict, 
        num_samples: int = 1, 
        sampler_params: dict = {},
        resolution: int = 64,
    ) -> torch.Tensor:
        """
        Sample sparse structure using TRELLIS.2 model.
        
        Returns coords compatible with TRELLIS1 slat flow model.
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, 
            noise, 
            **cond, 
            **sampler_params, 
            verbose=True
        ).samples
        
        decoder = self.models['sparse_structure_decoder']
        decoded = decoder(z_s) > 0
        
        # Adjust resolution if needed
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            if ratio > 1:
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def decode_slat(self, slat: sp.SparseTensor, formats: List[str] = ['gaussian']) -> dict:
        ret = {}
        if 'gaussian' in formats and 'slat_decoder_gs' in self.models:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret
    
    def sample_slat(self, cond: dict, coords: torch.Tensor, sampler_params: dict = {}) -> sp.SparseTensor:
        """Sample structured latent using TRELLIS1 model."""
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    def select_smallest_coords(self, coords: torch.Tensor, num_select: int = 1) -> torch.Tensor:
        """
        Select sparse structures with fewest voxels (like TRELLIS1 select_coords).

        Args:
            coords: [N, 4] tensor with [batch_idx, x, y, z]
            num_select: Number of structures to select (default 1)

        Returns:
            coords for the selected structure(s) with batch_idx reset
        """
        if coords.shape[0] == 0:
            return coords

        # Count voxels per batch
        batch_indices = coords[:, 0]
        unique_batches = batch_indices.unique()

        if len(unique_batches) <= num_select:
            # Not enough samples to select from, return all
            return coords

        # Count voxels for each batch
        counts = []
        for batch_idx in unique_batches:
            count = (batch_indices == batch_idx).sum().item()
            counts.append((batch_idx.item(), count))

        # Sort by count (ascending) and select smallest
        counts.sort(key=lambda x: x[1])
        selected_batches = [c[0] for c in counts[:num_select]]

        # Log selection
        for i, (batch_idx, count) in enumerate(counts):
            marker = "→" if batch_idx in selected_batches else " "
            print(f"  {marker} Sample {batch_idx}: {count} voxels")

        # Filter coords to selected batches only
        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        for batch_idx in selected_batches:
            mask |= (batch_indices == batch_idx)

        selected_coords = coords[mask].clone()

        # Reset batch indices to 0, 1, 2, ... for the selected samples
        if num_select == 1:
            selected_coords[:, 0] = 0
        else:
            # Remap batch indices
            for new_idx, old_idx in enumerate(selected_batches):
                selected_coords[selected_coords[:, 0] == old_idx, 0] = new_idx

        return selected_coords

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian'],
        preprocess_image: bool = True,
        resolution: int = 64,
        num_oversamples: int = 1,
    ) -> dict:
        """
        Run the hybrid pipeline with proper oversampling.

        Args:
            image: Input image
            num_samples: Number of final samples to return
            seed: Random seed
            sparse_structure_sampler_params: Override params for sparse structure sampling
            slat_sampler_params: Override params for slat sampling
            formats: Output formats (only 'gaussian' supported)
            preprocess_image: Whether to preprocess (remove background, crop)
            resolution: Target resolution for sparse structure
            num_oversamples: Generate N sparse structures, select smallest (like v12)

        Returns:
            dict with 'gaussian' key containing list of Gaussian objects
        """
        if preprocess_image:
            image = self.preprocess_image(image)

        # Get conditioning with DINOv3 for sparse structure (TRELLIS.2)
        cond_t2 = self.get_cond_t2([image], resolution=512)

        # Get conditioning with DINOv2 for slat (TRELLIS1) - THIS IS CRITICAL FOR COLORS!
        cond_t1 = self.get_cond_t1([image])

        # Effective oversamples = max(num_samples, num_oversamples)
        effective_oversamples = max(num_samples, num_oversamples)

        # Generate sparse structures sequentially (TRELLIS.2 doesn't support batching)
        # but select BEFORE running expensive SLat
        print(f"Sampling {effective_oversamples} sparse structures...")
        all_coords = []
        all_voxel_counts = []

        for i in range(effective_oversamples):
            torch.manual_seed(seed + i)
            coords_i = self.sample_sparse_structure(
                cond_t2,
                num_samples=1,
                sampler_params=sparse_structure_sampler_params,
                resolution=resolution,
            )
            voxel_count = coords_i.shape[0]
            all_coords.append(coords_i)
            all_voxel_counts.append(voxel_count)
            print(f"  Sample {i}: {voxel_count} voxels")

        # Select minimum by voxel count (cleanest geometry)
        if effective_oversamples > num_samples:
            # Sort by voxel count and select minimum
            sorted_indices = sorted(range(len(all_voxel_counts)), key=lambda i: all_voxel_counts[i])
            selected_idx = sorted_indices[0]  # minimum voxels
            print(f"  → Selected sample {selected_idx} with {all_voxel_counts[selected_idx]} voxels (min)")
            coords = all_coords[selected_idx]
        else:
            coords = all_coords[0]

        if coords.shape[0] == 0:
            print("  ⚠ No voxels generated!")
            return {'gaussian': []}

        # Sample SLat from TRELLIS1 using DINOv2 conditioning (runs ONCE on selected!)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond_t1, coords, slat_sampler_params)

        # Decode to Gaussian using TRELLIS1 decoder
        outputs = self.decode_slat(slat, formats)

        return {'gaussian': outputs.get('gaussian', [])}

    # =========================================================================
    # Multi-Image Support (V2)
    # =========================================================================

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name: The name of the sampler to inject ('sparse_structure_sampler' or 'slat_sampler').
            num_images: The number of images to condition on.
            num_steps: The number of steps to run the sampler for.
            mode: 'stochastic' cycles through images, 'multidiffusion' averages predictions.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, "_old_inference_model", sampler._inference_model)

        # Determine sampler type based on sampler_name
        is_t2_sampler = (sampler_name == "sparse_structure_sampler")

        if mode == "multidiffusion":
            if is_t2_sampler:
                # TRELLIS.2 sampler uses guidance_strength, guidance_interval (no neg_cond in signature)
                from .samplers_t2 import FlowEulerSampler as FlowEulerSamplerT2

                def _new_inference_model(
                    self,
                    model,
                    x_t,
                    t,
                    cond,
                    guidance_strength,
                    guidance_interval,
                    **kwargs,
                ):
                    neg_cond = kwargs.pop('neg_cond', None)
                    if guidance_interval[0] <= t <= guidance_interval[1] and guidance_strength != 1:
                        preds = []
                        for i in range(len(cond)):
                            preds.append(
                                FlowEulerSamplerT2._inference_model(
                                    self, model, x_t, t, cond[i : i + 1], **kwargs
                                )
                            )
                        pred_pos = sum(preds) / len(preds)
                        if neg_cond is not None:
                            pred_neg = FlowEulerSamplerT2._inference_model(
                                self, model, x_t, t, neg_cond, **kwargs
                            )
                            return guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
                        return pred_pos
                    else:
                        preds = []
                        for i in range(len(cond)):
                            preds.append(
                                FlowEulerSamplerT2._inference_model(
                                    self, model, x_t, t, cond[i : i + 1], **kwargs
                                )
                            )
                        return sum(preds) / len(preds)
            else:
                # TRELLIS1 sampler uses neg_cond, cfg_strength, cfg_interval
                from .samplers import FlowEulerSampler

                def _new_inference_model(
                    self,
                    model,
                    x_t,
                    t,
                    cond,
                    neg_cond,
                    cfg_strength,
                    cfg_interval,
                    **kwargs,
                ):
                    if cfg_interval[0] <= t <= cfg_interval[1]:
                        preds = []
                        for i in range(len(cond)):
                            preds.append(
                                FlowEulerSampler._inference_model(
                                    self, model, x_t, t, cond[i : i + 1], **kwargs
                                )
                            )
                        pred = sum(preds) / len(preds)
                        neg_pred = FlowEulerSampler._inference_model(
                            self, model, x_t, t, neg_cond, **kwargs
                        )
                        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                    else:
                        preds = []
                        for i in range(len(cond)):
                            preds.append(
                                FlowEulerSampler._inference_model(
                                    self, model, x_t, t, cond[i : i + 1], **kwargs
                                )
                            )
                        return sum(preds) / len(preds)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Only 'multidiffusion' is supported.")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, "_old_inference_model")

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian'],
        preprocess_image: bool = False,
        resolution: int = 64,
        num_oversamples: int = 1,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    ) -> dict:
        """
        Run the hybrid pipeline with multiple images as condition (multi-view).

        Args:
            images: List of input images (e.g., front, back, side views)
            num_samples: Number of final samples to return
            seed: Random seed
            sparse_structure_sampler_params: Override params for sparse structure sampling
            slat_sampler_params: Override params for slat sampling
            formats: Output formats (only 'gaussian' supported)
            preprocess_image: Whether to preprocess images
            resolution: Target resolution for sparse structure
            num_oversamples: Generate N sparse structures, select smallest
            mode: 'stochastic' or 'multidiffusion' for multi-image conditioning

        Returns:
            dict with 'gaussian' key containing list of Gaussian objects
        """
        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        num_images = len(images)
        print(f"Running multi-image pipeline with {num_images} views, mode={mode}")

        # Get conditioning for all images
        # DINOv3 for sparse structure (TRELLIS.2)
        cond_t2 = self.get_cond_t2(images, resolution=512)
        cond_t2['neg_cond'] = cond_t2['neg_cond'][:1]  # Only need one neg_cond

        # DINOv2 for slat (TRELLIS1) - CRITICAL for colors!
        cond_t1 = self.get_cond_t1(images)
        cond_t1['neg_cond'] = cond_t1['neg_cond'][:1]  # Only need one neg_cond

        # Effective oversamples
        effective_oversamples = max(num_samples, num_oversamples)

        # Get step counts for injection
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps', 12)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps', 12)

        print(f"Sampling {effective_oversamples} sparse structures with multi-image conditioning...")
        all_coords = []
        all_voxel_counts = []

        for i in range(effective_oversamples):
            torch.manual_seed(seed + i)

            # Use multi-image injection for sparse structure sampling
            with self.inject_sampler_multi_image(
                "sparse_structure_sampler", num_images, ss_steps, mode=mode
            ):
                coords_i = self.sample_sparse_structure(
                    cond_t2,
                    num_samples=1,
                    sampler_params=sparse_structure_sampler_params,
                    resolution=resolution,
                )

            voxel_count = coords_i.shape[0]
            all_coords.append(coords_i)
            all_voxel_counts.append(voxel_count)
            print(f"  Sample {i}: {voxel_count} voxels")

        # Select minimum by voxel count (cleanest geometry)
        if effective_oversamples > num_samples:
            sorted_indices = sorted(range(len(all_voxel_counts)), key=lambda i: all_voxel_counts[i])
            selected_idx = sorted_indices[0]
            print(f"  → Selected sample {selected_idx} with {all_voxel_counts[selected_idx]} voxels (min)")
            coords = all_coords[selected_idx]
        else:
            coords = all_coords[0]

        if coords.shape[0] == 0:
            print("  ⚠ No voxels generated!")
            return {'gaussian': []}

        # Sample SLat with multi-image conditioning
        torch.manual_seed(seed)
        with self.inject_sampler_multi_image(
            "slat_sampler", num_images, slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond_t1, coords, slat_sampler_params)

        # Decode to Gaussian
        outputs = self.decode_slat(slat, formats)

        return {'gaussian': outputs.get('gaussian', [])}

