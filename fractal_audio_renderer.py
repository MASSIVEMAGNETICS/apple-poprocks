"""
Fractal Audio Renderer
Enterprise-grade audio rendering at arbitrary length using fractal encoding.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from scipy.io import wavfile
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio rendering."""
    base_duration: float = 1.0
    sample_rate: int = 44100
    device: str = 'cpu'
    bit_depth: int = 16
    normalize: bool = True
    noise_scale: float = 0.1


class FractalAudioRenderer:
    """Renders audio at arbitrary length using fractal encoding."""

    def __init__(
        self,
        base_duration: float = 1.0,
        sample_rate: int = 44100,
        device: str = 'cpu',
        config: Optional[AudioConfig] = None
    ):
        """
        Initialize the fractal audio renderer.

        Args:
            base_duration: Base duration in seconds.
            sample_rate: Audio sample rate in Hz.
            device: Device to run computations on.
            config: Optional AudioConfig object.
        """
        if config is not None:
            self.base_duration = config.base_duration
            self.sample_rate = config.sample_rate
            self.device = torch.device(config.device)
            self.config = config
        else:
            self.base_duration = base_duration
            self.sample_rate = sample_rate
            self.device = torch.device(device)
            self.config = AudioConfig(
                base_duration=base_duration,
                sample_rate=sample_rate,
                device=device
            )

        # Validate parameters
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.base_duration <= 0:
            raise ValueError("base_duration must be positive")
        if self.config.bit_depth not in [8, 16, 24, 32]:
            raise ValueError("bit_depth must be 8, 16, 24, or 32")

        logger.info(f"Initialized FractalAudioRenderer")
        logger.info(f"Sample rate: {self.sample_rate} Hz")
        logger.info(f"Base duration: {self.base_duration} s")
        logger.info(f"Device: {self.device}")

    def interpolate_to_duration(
        self,
        base_audio: torch.Tensor,
        target_duration: float
    ) -> torch.Tensor:
        """
        Interpolate base audio to target duration using fractal principles.

        Args:
            base_audio: Base audio tensor.
            target_duration: Target duration in seconds.

        Returns:
            Interpolated audio tensor.
        """
        if not isinstance(base_audio, torch.Tensor):
            raise TypeError("base_audio must be a torch.Tensor")
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        base_samples = len(base_audio)
        target_samples = int(target_duration * self.sample_rate)

        if target_samples <= 0:
            raise ValueError("Invalid target duration")

        logger.debug(f"Interpolating from {base_samples} to {target_samples} samples")

        # Use linear interpolation as base
        indices = torch.linspace(
            0,
            base_samples - 1,
            target_samples,
            device=self.device
        )

        try:
            interpolated = torch.tensor(
                np.interp(
                    indices.cpu().numpy(),
                    np.arange(base_samples, dtype=np.float32),
                    base_audio.cpu().numpy()
                ),
                dtype=torch.float32,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            raise

        # Add fractal detail based on duration
        duration_factor = target_duration / self.base_duration

        if duration_factor > 1:
            # Generate fractal noise at higher duration
            noise = torch.randn(target_samples, device=self.device) * (
                1.0 / duration_factor
            )
            interpolated = interpolated + noise * self.config.noise_scale

            logger.debug(f"Added fractal noise (scale: {self.config.noise_scale})")

        # Clamp to valid range
        if self.config.normalize:
            interpolated = torch.clamp(interpolated, -1, 1)

        return interpolated

    def render_duration_level(
        self,
        base_audio: torch.Tensor,
        duration_multiplier: int
    ) -> torch.Tensor:
        """
        Render at specific duration level.

        Args:
            base_audio: Base audio tensor.
            duration_multiplier: Duration multiplier.

        Returns:
            Rendered audio tensor.
        """
        if not isinstance(base_audio, torch.Tensor):
            raise TypeError("base_audio must be a torch.Tensor")
        if not isinstance(duration_multiplier, int) or duration_multiplier <= 0:
            raise ValueError("duration_multiplier must be a positive integer")

        target_duration = self.base_duration * duration_multiplier

        logger.info(f"Rendering audio: {target_duration:.2f}s "
                   f"(multiplier: {duration_multiplier})")

        return self.interpolate_to_duration(base_audio, target_duration)

    def save_audio(
        self,
        tensor: torch.Tensor,
        filename: Union[str, Path],
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Save tensor as WAV file.

        Args:
            tensor: Audio tensor to save.
            filename: Output file path.
            sample_rate: Sample rate (uses config default if None).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = sample_rate if sample_rate is not None else self.sample_rate

        logger.info(f"Saving audio to {filename}")
        logger.info(f"Duration: {len(tensor)/sample_rate:.2f}s")

        try:
            # Ensure values are in [-1, 1]
            if self.config.normalize:
                max_val = torch.max(torch.abs(tensor))
                if max_val > 0:
                    tensor = tensor / max_val

            tensor = torch.clamp(tensor, -1, 1)

            # Convert to appropriate integer type based on bit depth
            if self.config.bit_depth == 8:
                # Map [-1, 1] -> [0, 255] for unsigned 8-bit WAV
                audio_array = ((tensor.cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
            elif self.config.bit_depth == 16:
                audio_array = (tensor.cpu().numpy() * 32767).astype(np.int16)
            elif self.config.bit_depth == 24:
                # 24-bit requires special handling
                audio_array = (tensor.cpu().numpy() * 8388607).astype(np.int32)
                # Write as 24-bit WAV (requires custom handling)
                self._save_24bit_wav(filename, audio_array, sample_rate)
                return
            elif self.config.bit_depth == 32:
                audio_array = (tensor.cpu().numpy() * 2147483647).astype(np.int32)
            else:
                raise ValueError(f"Unsupported bit depth: {self.config.bit_depth}")

            # Save as WAV
            wavfile.write(filename, sample_rate, audio_array)

            logger.info(f"Audio saved successfully: {filename}")
            logger.info(f"Bit depth: {self.config.bit_depth}-bit")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def _save_24bit_wav(
        self,
        filename: Path,
        audio_array: np.ndarray,
        sample_rate: int
    ) -> None:
        """
        Save 24-bit WAV file (special handling required).

        Args:
            filename: Output file path.
            audio_array: Audio data as int32.
            sample_rate: Sample rate.
        """
        import wave
        import struct

        with wave.open(str(filename), 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(3)  # 24-bit = 3 bytes
            wav_file.setframerate(sample_rate)

            # Convert to 24-bit bytes
            for sample in audio_array:
                # Convert numpy int32 to Python int before using to_bytes
                sample_bytes = int(sample).to_bytes(3, byteorder='little', signed=True)
                wav_file.writeframes(sample_bytes)

        logger.info("24-bit WAV saved successfully")

    def generate_tone(
        self,
        frequency: float,
        duration: float,
        wave_type: str = 'sine'
    ) -> torch.Tensor:
        """
        Generate a tone of specified frequency and duration.

        Args:
            frequency: Frequency in Hz.
            duration: Duration in seconds.
            wave_type: Type of wave ('sine', 'square', 'sawtooth', 'triangle').

        Returns:
            Audio tensor.
        """
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")

        n_samples = int(duration * self.sample_rate)
        t = torch.linspace(0, duration, n_samples, device=self.device)

        if wave_type == 'sine':
            audio = torch.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            audio = torch.sign(torch.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            audio = 2 * (t * frequency - torch.floor(t * frequency + 0.5))
        elif wave_type == 'triangle':
            audio = 2 * torch.abs(2 * (t * frequency - torch.floor(t * frequency + 0.5))) - 1
        else:
            raise ValueError(f"Unsupported wave type: {wave_type}")

        logger.debug(f"Generated {wave_type} wave: {frequency}Hz, {duration}s")

        return audio

    def apply_effects(
        self,
        audio: torch.Tensor,
        effects: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply audio effects.

        Args:
            audio: Input audio tensor.
            effects: Dictionary of effects to apply.

        Returns:
            Processed audio tensor.
        """
        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")

        processed = audio.clone()

        # Apply fade in/out
        if 'fade_in' in effects:
            fade_samples = int(effects['fade_in'] * self.sample_rate)
            fade_in = torch.linspace(0, 1, fade_samples, device=self.device)
            processed[:fade_samples] *= fade_in

        if 'fade_out' in effects:
            fade_samples = int(effects['fade_out'] * self.sample_rate)
            fade_out = torch.linspace(1, 0, fade_samples, device=self.device)
            processed[-fade_samples:] *= fade_out

        # Apply gain
        if 'gain' in effects:
            processed *= effects['gain']

        # Apply reverb (simple delay-based)
        if 'reverb' in effects:
            delay_samples = int(effects['reverb']['delay'] * self.sample_rate)
            decay = effects['reverb'].get('decay', 0.5)

            if delay_samples > 0:
                reverb = torch.zeros_like(processed)
                reverb[delay_samples:] += processed[:-delay_samples] * decay
                processed = processed + reverb

        logger.debug(f"Applied effects: {list(effects.keys())}")

        return processed

    def batch_render(
        self,
        base_audios: torch.Tensor,
        duration_multiplier: int
    ) -> torch.Tensor:
        """
        Render multiple audio clips at once.

        Args:
            base_audios: Batch of base audio tensors [B, T].
            duration_multiplier: Duration multiplier.

        Returns:
            Batch of rendered audio tensors [B, T'].
        """
        if not isinstance(base_audios, torch.Tensor):
            raise TypeError("base_audios must be a torch.Tensor")

        batch_size = base_audios.shape[0]
        rendered = []

        logger.info(f"Batch rendering {batch_size} audio clips")

        for i in range(batch_size):
            rendered_audio = self.render_duration_level(
                base_audios[i],
                duration_multiplier
            )
            rendered.append(rendered_audio)

        # Pad to same length if necessary
        max_length = max(len(r) for r in rendered)
        padded = []
        for r in rendered:
            if len(r) < max_length:
                padding = torch.zeros(max_length - len(r), device=self.device)
                r = torch.cat([r, padding])
            padded.append(r)

        return torch.stack(padded, dim=0)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    renderer = FractalAudioRenderer()

    # Generate A4 note (440 Hz)
    base_audio = renderer.generate_tone(440, 1.0, wave_type='sine')

    # Extend to 2 seconds
    extended = renderer.render_duration_level(base_audio, duration_multiplier=2)

    # Apply effects
    processed = renderer.apply_effects(
        extended,
        {'fade_in': 0.1, 'fade_out': 0.1, 'gain': 0.8}
    )

    # Save
    renderer.save_audio(processed, "output/extended_a4.wav")

    print(f"Extended audio shape: {processed.shape}")
    print(f"Duration: {len(processed)/renderer.sample_rate:.2f}s")
