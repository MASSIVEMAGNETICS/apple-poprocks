"""
Fractal Video Renderer
Enterprise-grade video rendering at arbitrary resolution and duration.
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from PIL import Image
import imageio
import logging
from dataclasses import dataclass
from pathlib import Path
import tempfile

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video rendering."""
    base_resolution: Tuple[int, int] = (16, 16)
    base_duration: int = 10
    device: str = 'cpu'
    fps: int = 30
    codec: str = 'libx264'
    quality: float = 0.8
    output_format: str = 'mp4'


class FractalVideoRenderer:
    """Renders video at arbitrary resolution and duration using fractal encoding."""

    def __init__(
        self,
        base_resolution: Tuple[int, int] = (16, 16),
        base_duration: int = 10,
        device: str = 'cpu',
        config: Optional[VideoConfig] = None
    ):
        """
        Initialize the fractal video renderer.

        Args:
            base_resolution: Base resolution tuple (height, width).
            base_duration: Base duration in frames.
            device: Device to run computations on.
            config: Optional VideoConfig object.
        """
        if config is not None:
            self.base_resolution = config.base_resolution
            self.base_duration = config.base_duration
            self.device = torch.device(config.device)
            self.config = config
        else:
            self.base_resolution = base_resolution
            self.base_duration = base_duration
            self.device = torch.device(device)
            self.config = VideoConfig(
                base_resolution=base_resolution,
                base_duration=base_duration,
                device=device
            )

        self.fractal_renderer = None  # Will be initialized with first call

        # Validate parameters
        if len(self.base_resolution) != 2:
            raise ValueError("base_resolution must be a tuple of (height, width)")
        if self.base_resolution[0] <= 0 or self.base_resolution[1] <= 0:
            raise ValueError("Resolution dimensions must be positive")
        if self.config.fps <= 0:
            raise ValueError("fps must be positive")

        logger.info(f"Initialized FractalVideoRenderer")
        logger.info(f"Base resolution: {self.base_resolution}")
        logger.info(f"Base duration: {self.base_duration} frames")
        logger.info(f"Device: {self.device}")

    def initialize_renderer(self) -> None:
        """Initialize fractal renderer."""
        if self.fractal_renderer is None:
            try:
                from fractal_renderer import FractalRenderer
                self.fractal_renderer = FractalRenderer(
                    base_resolution=self.base_resolution,
                    device=str(self.device)
                )
                logger.info("Fractal renderer initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import FractalRenderer: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize fractal renderer: {e}")
                raise

    def render_frame(
        self,
        base_frame: torch.Tensor,
        zoom_level: int
    ) -> torch.Tensor:
        """
        Render single frame at zoom level.

        Args:
            base_frame: Base frame tensor.
            zoom_level: Zoom level (power of 2).

        Returns:
            Rendered frame tensor.
        """
        if not isinstance(base_frame, torch.Tensor):
            raise TypeError("base_frame must be a torch.Tensor")
        if not isinstance(zoom_level, int) or zoom_level < 0:
            raise ValueError("zoom_level must be a non-negative integer")

        self.initialize_renderer()

        try:
            rendered_frame = self.fractal_renderer.render_zoom_level(
                base_frame, zoom_level
            )
            logger.debug(f"Rendered frame at zoom level {zoom_level}: "
                        f"{rendered_frame.shape}")
            return rendered_frame
        except Exception as e:
            logger.error(f"Failed to render frame: {e}")
            raise

    def render_video(
        self,
        base_frames: List[torch.Tensor],
        zoom_level: int,
        fps: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Render entire video at zoom level.

        Args:
            base_frames: List of base frame tensors.
            zoom_level: Zoom level.
            fps: Frames per second (uses config default if None).

        Returns:
            List of rendered frame tensors.
        """
        if not isinstance(base_frames, list):
            raise TypeError("base_frames must be a list")
        if len(base_frames) == 0:
            raise ValueError("base_frames cannot be empty")

        fps = fps if fps is not None else self.config.fps

        logger.info(f"Rendering video: {len(base_frames)} frames at zoom level {zoom_level}")

        rendered_frames = []

        for idx, frame in enumerate(base_frames):
            try:
                rendered_frame = self.render_frame(frame, zoom_level)
                rendered_frames.append(rendered_frame)

                if (idx + 1) % max(1, len(base_frames) // 10) == 0:
                    logger.debug(f"Progress: {idx + 1}/{len(base_frames)} frames")

            except Exception as e:
                logger.error(f"Failed to render frame {idx}: {e}")
                raise

        logger.info(f"Video rendering completed: {len(rendered_frames)} frames")

        return rendered_frames

    def save_video(
        self,
        frames: List[torch.Tensor],
        filename: Union[str, Path],
        fps: Optional[int] = None,
        codec: Optional[str] = None
    ) -> None:
        """
        Save frames as video file.

        Args:
            frames: List of frame tensors.
            filename: Output file path.
            fps: Frames per second.
            codec: Video codec.
        """
        if not isinstance(frames, list) or len(frames) == 0:
            raise ValueError("frames must be a non-empty list")

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        fps = fps if fps is not None else self.config.fps
        codec = codec if codec is not None else self.config.codec

        logger.info(f"Saving video to {filename} at {fps} fps")

        try:
            # Convert to PIL Images
            pil_frames = []
            for idx, frame in enumerate(frames):
                pil_img = self.tensor_to_pil(frame)
                pil_frames.append(pil_img)

            # Determine format based on extension
            format_map = {
                '.mp4': 'mp4',
                '.avi': 'avi',
                '.gif': 'gif',
                '.webm': 'webm'
            }

            ext = filename.suffix.lower()
            output_format = format_map.get(ext, 'mp4')

            # Save video
            if ext == '.gif':
                imageio.mimsave(filename, pil_frames, fps=fps, format=output_format)
            else:
                imageio.mimsave(
                    filename,
                    pil_frames,
                    fps=fps,
                    format=output_format,
                    codec=codec,
                    quality=self.config.quality
                )

            logger.info(f"Video saved successfully: {filename}")
            logger.info(f"Total frames: {len(frames)}, Duration: {len(frames)/fps:.2f}s")

        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image.

        Args:
            tensor: Tensor to convert.

        Returns:
            PIL Image object.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        if tensor.dim() == 2:  # Grayscale
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)  # Make RGB
        elif tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)  # CHW to HWC

        # Ensure values are in [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to uint8
        img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def batch_render_video(
        self,
        base_frames: torch.Tensor,
        zoom_level: int,
        fps: Optional[int] = None,
        batch_size: int = 8
    ) -> List[torch.Tensor]:
        """
        Render video using batch processing for efficiency.

        Args:
            base_frames: Batch of base frames [B, H, W] or [B, H, W, C].
            zoom_level: Zoom level.
            fps: Frames per second.
            batch_size: Processing batch size.

        Returns:
            List of rendered frames.
        """
        if not isinstance(base_frames, torch.Tensor):
            raise TypeError("base_frames must be a torch.Tensor")

        self.initialize_renderer()

        n_frames = base_frames.shape[0]
        rendered_frames = []

        logger.info(f"Batch rendering {n_frames} frames with batch_size {batch_size}")

        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch = base_frames[start_idx:end_idx]

            # Use batch render from fractal renderer
            rendered_batch = self.fractal_renderer.batch_render(batch, zoom_level)

            for i in range(rendered_batch.shape[0]):
                rendered_frames.append(rendered_batch[i])

            logger.debug(f"Processed batch {start_idx//batch_size + 1}: "
                        f"{end_idx}/{n_frames} frames")

        return rendered_frames

    def create_preview(
        self,
        frames: List[torch.Tensor],
        preview_size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        Create a preview grid of video frames.

        Args:
            frames: List of frame tensors.
            preview_size: Size of preview grid.

        Returns:
            PIL Image with frame grid.
        """
        if not frames:
            raise ValueError("frames cannot be empty")

        # Select representative frames (first, middle, last, etc.)
        n_preview = min(9, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_preview, dtype=int)

        # Create grid
        grid_size = int(np.ceil(np.sqrt(n_preview)))

        preview_frames = []
        for idx in indices:
            frame = frames[idx]
            # Resize for preview
            if frame.dim() == 2:
                frame = frame.unsqueeze(-1).repeat(1, 1, 3)

            frame_resized = torch.nn.functional.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(preview_size[0] // grid_size, preview_size[1] // grid_size),
                mode='bilinear',
                align_corners=True
            ).squeeze(0).permute(1, 2, 0)

            preview_frames.append(frame_resized)

        # Arrange in grid
        cell_width = preview_size[1] // grid_size
        full_row_width = grid_size * cell_width
        rows = []
        for i in range(grid_size):
            row_frames = preview_frames[i*grid_size:(i+1)*grid_size]
            if not row_frames:
                break
            row = torch.cat(row_frames, dim=1)
            # Pad row to full width if this row has fewer frames than grid_size
            if row.shape[1] < full_row_width:
                pad = torch.zeros(row.shape[0], full_row_width - row.shape[1], row.shape[2],
                                  device=row.device)
                row = torch.cat([row, pad], dim=1)
            rows.append(row)

        if not rows:
            raise ValueError("No frames to create preview")

        grid = torch.cat(rows, dim=0)

        return self.tensor_to_pil(grid)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    renderer = FractalVideoRenderer()
    base_frames = [torch.rand(16, 16, 3) for _ in range(10)]

    rendered = renderer.render_video(base_frames, zoom_level=2, fps=30)

    # Save video
    renderer.save_video(rendered, "output/test_video.mp4")

    print(f"Rendered {len(rendered)} frames")
    print(f"Frame shape: {rendered[0].shape}")
