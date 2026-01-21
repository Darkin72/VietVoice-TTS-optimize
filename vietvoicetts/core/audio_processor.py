"""
Audio processing utilities for TTS inference
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from typing import List
import io


class AudioProcessor:
    """Handles audio processing operations"""

    @staticmethod
    def load_audio(path_or_bytes: str | bytes, sample_rate: int) -> np.ndarray:
        """Load and process audio file, returns int16 normalized audio (Matching reference)"""
        if isinstance(path_or_bytes, str):
            if not Path(path_or_bytes).exists():
                raise FileNotFoundError(f"Audio file not found: {path_or_bytes}")
            audio_segment = (
                AudioSegment.from_file(path_or_bytes)
                .set_channels(1)
                .set_frame_rate(sample_rate)
            )
        else:
            audio_segment = (
                AudioSegment.from_file(io.BytesIO(path_or_bytes))
                .set_channels(1)
                .set_frame_rate(sample_rate)
            )

        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        return AudioProcessor.normalize_to_int16(samples)

    @staticmethod
    def normalize_to_int16(audio: np.ndarray) -> np.ndarray:
        """Scale float32 audio to int16 range (Matching reference)"""
        if audio.dtype == np.int16:
            return audio

        # Remove DC offset
        audio = audio - np.mean(audio)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Scale to 29491 (Standard for VietVoice reference)
            scaling_factor = 29491.0 / max_val
            normalized_audio = audio * scaling_factor
        else:
            normalized_audio = audio

        return np.clip(normalized_audio, -32768, 32767).astype(np.int16)

    @staticmethod
    def fix_clipped_audio(audio: np.ndarray) -> np.ndarray:
        """Reduce volume if audio is clipping (Handles both float32 and int16)"""
        if len(audio) == 0:
            return audio

        max_val = np.max(np.abs(audio))

        # Determine if we are in int16 range or float32 range
        is_int_range = max_val > 1.5
        threshold = 32767.0 if is_int_range else 0.99
        target = 26214.0 if is_int_range else 0.8

        if max_val >= threshold:
            scale_factor = target / max_val
            return audio * scale_factor
        return audio

    @staticmethod
    def to_int16_safe(audio: np.ndarray) -> np.ndarray:
        """
        Convert any audio signal to int16 safely for streaming.
        Handles both [-1, 1] and [-32k, 32k] ranges without per-chunk normalization.
        """
        if audio.dtype == np.int16:
            return audio

        if len(audio) == 0:
            return audio.astype(np.int16)

        max_val = np.max(np.abs(audio))

        # If the values are very small, they are likely in [-1, 1] range
        if max_val <= 1.5:
            # Scale to standard int16 magnitude and round
            return np.clip(np.round(audio * 29491.0), -32768, 32767).astype(np.int16)

        # Already in large range, just clip and round
        return np.clip(np.round(audio), -32768, 32767).astype(np.int16)

    @staticmethod
    def concatenate_with_crossfade_improved(
        generated_waves: List[np.ndarray], cross_fade_duration: float, sample_rate: int
    ) -> np.ndarray:
        """Improved concatenation with volume matching (Matching reference)"""
        if not generated_waves:
            return np.array([])

        if len(generated_waves) == 1:
            return generated_waves[0].reshape(-1)

        # Check if we are dealing with float or int audio
        # Note: if it's float, we expect values in [-1, 1] or similar
        first_wave = generated_waves[0].reshape(-1)
        is_int_audio = first_wave.dtype == np.int16 or np.max(np.abs(first_wave)) > 1.5
        threshold = 100 if is_int_audio else 0.003

        flattened_waves = []
        for wave in generated_waves:
            flat_wave = wave.reshape(-1)
            fixed_wave = AudioProcessor.fix_clipped_audio(flat_wave)
            flattened_waves.append(fixed_wave)

        if cross_fade_duration <= 0:
            return np.concatenate(flattened_waves)

        final_wave = flattened_waves[0]

        for i in range(1, len(flattened_waves)):
            prev_wave = final_wave
            next_wave = flattened_waves[i]

            cross_fade_samples = int(cross_fade_duration * sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            prev_rms = np.sqrt(np.mean(prev_overlap.astype(np.float32) ** 2))
            next_rms = np.sqrt(np.mean(next_overlap.astype(np.float32) ** 2))

            if prev_rms > threshold and next_rms > threshold:
                volume_ratio = prev_rms / next_rms
                volume_ratio = np.clip(volume_ratio, 0.7, 1.5)
                next_wave_adjusted = next_wave.astype(np.float32) * volume_ratio
            else:
                next_wave_adjusted = next_wave

            fade_out = np.cos(np.linspace(0, np.pi / 2, cross_fade_samples)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, cross_fade_samples)) ** 2

            cross_faded_overlap = (
                prev_overlap.astype(np.float32) * fade_out
                + next_overlap[:cross_fade_samples].astype(np.float32) * fade_in
            )

            # Match output type to input type
            if is_int_audio:
                cross_faded_overlap = cross_faded_overlap.astype(np.int16)
                if next_wave_adjusted.dtype != np.int16:
                    next_wave_adjusted = next_wave_adjusted.astype(np.int16)

            final_wave = np.concatenate(
                [
                    prev_wave[:-cross_fade_samples],
                    cross_faded_overlap,
                    next_wave_adjusted[cross_fade_samples:],
                ]
            )

        return final_wave
