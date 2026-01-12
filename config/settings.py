"""
Configuration settings for TRELLIS-HYBRID-V2 (Multi-View).

Pipeline: Image -> Qwen Multi-View (3 views) -> BiRefNet (each) -> TRELLIS Multi-Image -> PLY
"""

from pathlib import Path
from typing import Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    api_title: str = "TRELLIS2 Gaussian Splatting Generator"
    api_description: str = "Image to 3D Gaussian Splatting PLY generation"

    # ==========================================================================
    # API Settings
    # ==========================================================================
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=10006, env="PORT")

    # ==========================================================================
    # GPU Settings
    # ==========================================================================
    gpu: int = Field(default=0, env="GPU")
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")

    # ==========================================================================
    # Output Settings
    # ==========================================================================
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # ==========================================================================
    # TRELLIS Model Settings (Hybrid: TRELLIS2 sparse + TRELLIS1 SLat/GS)
    # ==========================================================================
    # TRELLIS1 model (for SLat flow + GS decoder)
    trellis1_model_id: str = Field(
        default="microsoft/TRELLIS-image-large",
        env="TRELLIS1_MODEL_ID"
    )

    # TRELLIS.2 model (for sparse structure)
    trellis2_model_id: str = Field(
        default="microsoft/TRELLIS.2-4B",
        env="TRELLIS2_MODEL_ID"
    )

    # Sparse structure sampler (tuned for quality)
    sparse_structure_steps: int = Field(default=12, env="SPARSE_STRUCTURE_STEPS")
    sparse_structure_cfg: float = Field(default=5.5, env="SPARSE_STRUCTURE_CFG")

    # SLat sampler (same as v12)
    slat_steps: int = Field(default=20, env="SLAT_STEPS")
    slat_cfg: float = Field(default=2.45, env="SLAT_CFG")

    # Oversampling (generates N variants, picks smallest)
    num_oversamples: int = Field(default=1, env="NUM_OVERSAMPLES")

    # ==========================================================================
    # Qwen Image Edit Settings (diffusers + LoRA)
    # ==========================================================================
    use_qwen_edit: bool = Field(default=True, env="USE_QWEN_EDIT")
    qwen_dtype: str = Field(default="bf16", env="QWEN_DTYPE")
    qwen_num_inference_steps: int = Field(default=4, env="QWEN_NUM_INFERENCE_STEPS")
    qwen_true_cfg_scale: float = Field(default=1.0, env="QWEN_TRUE_CFG_SCALE")
    qwen_height: int = Field(default=1024, env="QWEN_HEIGHT")
    qwen_width: int = Field(default=1024, env="QWEN_WIDTH")
    qwen_model_path: str = Field(
        default="Qwen/Qwen-Image-Edit-2509",
        env="QWEN_MODEL_PATH"
    )
    qwen_lora_path: str = Field(
        default="lightx2v/Qwen-Image-Lightning",
        env="QWEN_LORA_PATH"
    )
    qwen_lora_filename: str = Field(
        default="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
        env="QWEN_LORA_FILENAME"
    )

    # Qwen edit prompt config
    qwen_edit_prompt_path: Path = Field(
        default=config_dir / "qwen_edit_prompt.json",
        env="QWEN_EDIT_PROMPT_PATH"
    )

    # ==========================================================================
    # Background Removal Settings (BiRefNet only in V2)
    # ==========================================================================
    background_removal_model_id: str = Field(
        default="ZhengPeng7/BiRefNet",
        env="BACKGROUND_REMOVAL_MODEL_ID"
    )

    # ==========================================================================
    # Compression Settings
    # ==========================================================================
    compression: bool = Field(default=False, env="COMPRESSION")

    # ==========================================================================
    # Image Processing Settings (same as v12)
    # ==========================================================================
    input_image_size: Tuple[int, int] = Field(
        default=(1024, 1024),
        env="INPUT_IMAGE_SIZE"
    )
    output_image_size: Tuple[int, int] = Field(
        default=(518, 518),
        env="OUTPUT_IMAGE_SIZE"
    )
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()

__all__ = ["Settings", "settings"]
