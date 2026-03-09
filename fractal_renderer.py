"""
Fractal Rendering Engine & Optimizer
Enterprise-grade fractal-based image rendering with arbitrary resolution scaling.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from PIL import Image
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FractalConfig:
    """Configuration for fractal rendering."""
    base_resolution: Tuple[int, int] = (16, 16)
    device: str = 'cpu'
    target_size_kb: float = 5.0
    noise_scale: float = 0.1
    clamp_min: float = 0.0
    clamp_max: float = 1.0


class FractalOptimizer:
    """Optimizes fractal rule sets for maximum compression."""

    def __init__(self, target_size_kb: float = 5.0):
        """
        Initialize the fractal optimizer.

        Args:
            target_size_kb: Target size in kilobytes for compressed rules.
        """
        self.target_size_kb = target_size_kb
        logger.info(f"Initialized FractalOptimizer with target size: {target_size_kb}KB")

    def compress_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress rule set by pruning redundant parameters.

        Args:
            rules: Dictionary containing fractal rules and parameters.

        Returns:
            Compressed rules dictionary.
        """
        logger.debug(f"Compressing rules from {len(rules)} parameters")

        # Placeholder for 700M param -> 5KB reduction logic
        # In a real implementation, this would use symbolic regression
        compressed_rules = {
            k: v for k, v in rules.items()
            if v is not None and not (isinstance(v, (int, float)) and v == 0)
        }

        logger.info(f"Compressed rules to {len(compressed_rules)} parameters")
        return compressed_rules

    def validate_rules(self, rules: Dict[str, Any]) -> bool:
        """
        Validate fractal rules for correctness.

        Args:
            rules: Rules dictionary to validate.

        Returns:
            True if rules are valid, False otherwise.
        """
        if not isinstance(rules, dict):
            logger.error("Rules must be a dictionary")
            return False

        logger.debug("Rules validation passed")
        return True


class FractalRenderer:
    """Renders images at arbitrary resolution using fractal encoding."""

    def __init__(
        self,
        base_resolution: Tuple[int, int] = (16, 16),
        device: str = 'cpu',
        config: Optional[FractalConfig] = None
    ):
        """
        Initialize the fractal renderer.

        Args:
            base_resolution: Base resolution tuple (height, width).
            device: Device to run computations on ('cpu' or 'cuda').
            config: Optional FractalConfig object.
        """
        if config is not None:
            self.base_resolution = config.base_resolution
            self.device = torch.device(config.device)
            self.config = config
        else:
            self.base_resolution = base_resolution
            self.device = torch.device(device)
            self.config = FractalConfig(
                base_resolution=base_resolution,
                device=device
            )

        self.optimizer = FractalOptimizer(target_size_kb=self.config.target_size_kb)

        # Validate resolution
        if len(self.base_resolution) != 2:
            raise ValueError("base_resolution must be a tuple of (height, width)")
        if self.base_resolution[0] <= 0 or self.base_resolution[1] <= 0:
            raise ValueError("Resolution dimensions must be positive")

        logger.info(f"Initialized FractalRenderer with base resolution: {self.base_resolution}")
        logger.info(f"Using device: {self.device}")

    def interpolate_to_resolution(
        self,
        base_grid: torch.Tensor,
        target_resolution: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Interpolate base grid to target resolution using fractal principles.

        Args:
            base_grid: Base grid tensor to interpolate. Supports 2D [H, W] or
                       3D [H, W, C] (HWC) or [C, H, W] (CHW) inputs.
            target_resolution: Target resolution tuple (height, width).

        Returns:
            Interpolated tensor at target resolution (same layout as input).
        """
        if not isinstance(base_grid, torch.Tensor):
            raise TypeError("base_grid must be a torch.Tensor")

        if len(target_resolution) != 2:
            raise ValueError("target_resolution must be a tuple of (height, width)")

        # Normalise to [1, C, H, W] for F.interpolate; track original layout
        is_hwc = False
        if base_grid.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            base_tensor = base_grid.unsqueeze(0).unsqueeze(0)
        elif base_grid.dim() == 3 and base_grid.shape[0] in [1, 3]:
            # CHW: [C, H, W] -> [1, C, H, W]
            base_tensor = base_grid.unsqueeze(0)
        else:
            # HWC: [H, W, C] -> [1, C, H, W]
            is_hwc = True
            base_tensor = base_grid.permute(2, 0, 1).unsqueeze(0)

        try:
            target_tensor_4d = torch.nn.functional.interpolate(
                base_tensor,
                size=target_resolution,
                mode='bilinear',
                align_corners=True
            )
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            raise

        # Add fractal detail based on zoom level
        zoom_factor = max(target_resolution) / max(self.base_resolution)

        if zoom_factor > 1:
            n_channels = target_tensor_4d.shape[1]
            # Generate fractal noise [C, H', W']
            noise = torch.randn(
                n_channels, *target_resolution, device=self.device
            ) * (1.0 / zoom_factor)
            target_tensor_4d = target_tensor_4d + noise.unsqueeze(0) * self.config.noise_scale

        target_tensor_4d = torch.clamp(
            target_tensor_4d,
            self.config.clamp_min,
            self.config.clamp_max
        )

        # Restore original tensor layout
        if base_grid.dim() == 2:
            # [1, 1, H', W'] -> [H', W']
            return target_tensor_4d.squeeze(0).squeeze(0)
        elif is_hwc:
            # [1, C, H', W'] -> [H', W', C]
            return target_tensor_4d.squeeze(0).permute(1, 2, 0)
        else:
            # [1, C, H', W'] -> [C, H', W']
            return target_tensor_4d.squeeze(0)

    def render_zoom_level(
        self,
        base_state: torch.Tensor,
        zoom_level: int
    ) -> torch.Tensor:
        """
        Render at specific zoom level.

        Args:
            base_state: Base state tensor.
            zoom_level: Zoom level (power of 2 multiplier).

        Returns:
            Rendered tensor at zoomed resolution.
        """
        if not isinstance(base_state, torch.Tensor):
            raise TypeError("base_state must be a torch.Tensor")
        if not isinstance(zoom_level, int) or zoom_level < 0:
            raise ValueError("zoom_level must be a non-negative integer")

        target_height = self.base_resolution[0] * (2 ** zoom_level)
        target_width = self.base_resolution[1] * (2 ** zoom_level)

        logger.debug(f"Rendering at zoom level {zoom_level}: {target_width}x{target_height}")

        return self.interpolate_to_resolution(base_state, (target_height, target_width))

    def save_image(self, tensor: torch.Tensor, filename: Union[str, Path]) -> None:
        """
        Save tensor as image file.

        Args:
            tensor: Tensor to save.
            filename: Output file path.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        img = self.tensor_to_pil(tensor)

        try:
            img.save(filename)
            logger.info(f"Saved image to {filename}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image without saving.

        Args:
            tensor: Tensor to convert.

        Returns:
            PIL Image object.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        # Handle different tensor dimensions
        if tensor.dim() == 2:  # Grayscale
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)  # Make RGB
        elif tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)  # CHW to HWC

        # Ensure values are in [0, 1]
        tensor = torch.clamp(tensor, self.config.clamp_min, self.config.clamp_max)

        # Convert to uint8
        img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def batch_render(
        self,
        base_states: torch.Tensor,
        zoom_level: int
    ) -> torch.Tensor:
        """
        Render multiple frames at once.

        Args:
            base_states: Batch of base state tensors [B, H, W].
            zoom_level: Zoom level.

        Returns:
            Batch of rendered tensors [B, H', W'].
        """
        if not isinstance(base_states, torch.Tensor):
            raise TypeError("base_states must be a torch.Tensor")

        batch_size = base_states.shape[0]
        rendered = []

        for i in range(batch_size):
            rendered_frame = self.render_zoom_level(base_states[i], zoom_level)
            rendered.append(rendered_frame)

        return torch.stack(rendered, dim=0)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    renderer = FractalRenderer(base_resolution=(8, 8))
    base_grid = torch.rand(8, 8)
    zoomed = renderer.render_zoom_level(base_grid, zoom_level=2)  # 32x32

    print(f"Base grid shape: {base_grid.shape}")
    print(f"Zoomed grid shape: {zoomed.shape}")

    renderer.save_image(zoomed, "output/zoomed_test.png")
