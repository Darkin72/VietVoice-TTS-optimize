"""
Audio processing utilities for TTS inference
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
import io


class AudioProcessor:
    """Handles audio processing operations"""

    @staticmethod
    def load_audio(path_or_bytes: str | bytes, sample_rate: int) -> np.ndarray:
        """Load and process audio file"""
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
        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        return AudioProcessor.normalize_to_int16(audio)

    @staticmethod
    def normalize_to_int16(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to int16 range with proper scaling to prevent clipping"""
        # Remove DC offset
        audio = audio - np.mean(audio)

        # Get maximum absolute value
        max_val = np.max(np.abs(audio))

        if max_val > 0:
            # Use 90% of max range to prevent clipping and allow headroom
            scaling_factor = 29491.0 / max_val  # 90% of 32767
            normalized_audio = audio * scaling_factor
        else:
            normalized_audio = audio

        return normalized_audio.astype(np.int16)

    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sample_rate: int) -> None:
        """Save audio to file"""
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio.reshape(-1), sample_rate, format="WAVEX")
