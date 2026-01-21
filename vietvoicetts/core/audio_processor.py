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
        """Load and process audio file as float32 normalized to [-1.0, 1.0]"""
        if isinstance(path_or_bytes, str):
            if not Path(path_or_bytes).exists():
                raise FileNotFoundError(f"Audio file not found: {path_or_bytes}")
            audio_segment = AudioSegment.from_file(path_or_bytes).set_channels(1).set_frame_rate(sample_rate)
        else:
            audio_segment = AudioSegment.from_file(io.BytesIO(path_or_bytes)).set_channels(1).set_frame_rate(sample_rate)
        
        # Proper normalization to float32 [-1.0, 1.0]
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        # Pydub standard for 16-bit PCM is dividing by 32768
        max_possible_val = float(1 << (8 * audio_segment.sample_width - 1))
        return samples / max_possible_val
    
    @staticmethod
    def normalize_to_int16(audio: np.ndarray) -> np.ndarray:
        """Scale float32 audio to int16 range"""
        if audio.dtype == np.int16:
            return audio
            
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Scale to 90% of INT16 range
            scaling_factor = 29491.0 / max_val
            normalized_audio = audio * scaling_factor
        else:
            normalized_audio = audio
        
        return np.clip(normalized_audio, -32768, 32767).astype(np.int16)
    
    @staticmethod
    def fix_clipped_audio(audio: np.ndarray) -> np.ndarray:
        """Reduce volume if audio is clipping"""
        max_val = np.max(np.abs(audio))
        
        # Handle both float32 [-1, 1] and int16 ranges
        threshold = 32767.0 if max_val > 1.1 else 0.99
        
        if max_val >= threshold:
            scale_factor = (0.8 * threshold) / max_val
            return audio * scale_factor
        return audio
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sample_rate: int) -> None:
        """Save audio to file"""
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio.reshape(-1), sample_rate, format='WAVEX')
    
    @staticmethod
    def concatenate_with_crossfade(generated_waves: List[np.ndarray], 
                                   cross_fade_duration: float, 
                                   sample_rate: int) -> np.ndarray:
        """Concatenate multiple audio waves with cross-fading"""
        if not generated_waves:
            return np.array([])
        
        if len(generated_waves) == 1:
            return generated_waves[0].reshape(-1)  # Flatten to 1D
        
        # Flatten all waves to 1D arrays
        flattened_waves = [wave.reshape(-1) for wave in generated_waves]
        
        if cross_fade_duration <= 0:
            # Simply concatenate
            return np.concatenate(flattened_waves)
        
        # Combine all generated waves with cross-fading
        final_wave = flattened_waves[0]
        for i in range(1, len(flattened_waves)):
            prev_wave = final_wave
            next_wave = flattened_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

        return final_wave

    @staticmethod
    def concatenate_with_crossfade_improved(generated_waves: List[np.ndarray], 
                                           cross_fade_duration: float, 
                                           sample_rate: int) -> np.ndarray:
        """Improved concatenation for both float32 and int16 audio"""
        if not generated_waves:
            return np.array([])
        
        if len(generated_waves) == 1:
            return generated_waves[0].reshape(-1)
        
        # Check if we are dealing with float or int audio
        is_int_audio = generated_waves[0].dtype == np.int16
        threshold = 100 if is_int_audio else 0.001
        
        flattened_waves = []
        for wave in generated_waves:
            flat_wave = wave.reshape(-1)
            # Fix clipped audio (handles both float and int ranges)
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
                next_wave_adjusted = (next_wave.astype(np.float32) * volume_ratio)
                if is_int_audio:
                    next_wave_adjusted = next_wave_adjusted.astype(np.int16)
                next_overlap = next_wave_adjusted[:cross_fade_samples]
            else:
                next_wave_adjusted = next_wave
                next_overlap = next_wave_adjusted[:cross_fade_samples]

            fade_out = np.cos(np.linspace(0, np.pi/2, cross_fade_samples)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi/2, cross_fade_samples)) ** 2

            cross_faded_overlap = (prev_overlap.astype(np.float32) * fade_out + 
                                   next_overlap.astype(np.float32) * fade_in)
            if is_int_audio:
                cross_faded_overlap = cross_faded_overlap.astype(np.int16)

            final_wave = np.concatenate([
                prev_wave[:-cross_fade_samples], 
                cross_faded_overlap, 
                next_wave_adjusted[cross_fade_samples:]
            ])

        return final_wave

            # Combine waves
            final_wave = np.concatenate([
                prev_wave[:-cross_fade_samples], 
                cross_faded_overlap, 
                next_wave_adjusted[cross_fade_samples:]
            ])

        return final_wave 