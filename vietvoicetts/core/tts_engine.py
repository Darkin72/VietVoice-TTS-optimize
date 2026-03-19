"""
TTS Engine - Main speech synthesis engine
"""

import time
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm

from .model_config import ModelConfig
from .model import ModelSessionManager
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor


class TTSEngine:
    """Main TTS engine for inference"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model_session_manager = ModelSessionManager(self.config)
        self.model_session_manager.load_models()

        if not self.model_session_manager.vocab_path:
            raise RuntimeError("Vocabulary file not found in model tar archive")

        self.text_processor = TextProcessor(self.model_session_manager.vocab_path)
        self.audio_processor = AudioProcessor()
        self.sample_cache = {}

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.model_session_manager:
            self.model_session_manager.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _prepare_inputs(
        self, reference_audio_path_or_bytes: str, reference_text: str, target_text: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare single-pass inputs for inference."""
        audio = self.audio_processor.load_audio(
            reference_audio_path_or_bytes, self.config.sample_rate
        )
        audio = audio.reshape(1, 1, -1)

        # Clean text
        reference_text = self.text_processor.clean_text(reference_text)
        target_text = self.text_processor.clean_text(target_text)

        # Calculate reference audio duration and text length
        ref_text_len = self.text_processor.calculate_text_length(
            reference_text, self.config.pause_punctuation
        )
        ref_audio_len = audio.shape[-1] // self.config.hop_length + 1
        ref_audio_duration = audio.shape[-1] / self.config.sample_rate

        # Estimate speaking rate (characters per second)
        speaking_rate = (
            ref_text_len / ref_audio_duration if ref_audio_duration > 0 else 100
        )

        # Calculate total duration including reference audio
        target_text_len = self.text_processor.calculate_text_length(
            target_text, self.config.pause_punctuation
        )
        target_audio_duration = max(
            target_text_len / speaking_rate / self.config.speed,
            self.config.min_target_duration,
        )
        total_estimated_duration = ref_audio_duration + target_audio_duration

        print(
            "Server-side chunking disabled. "
            f"Processing single chunk with estimated total duration {total_estimated_duration:.1f}s "
            f"(ref: {ref_audio_duration:.1f}s + target: {target_audio_duration:.1f}s)."
        )

        target_audio_samples = int(target_audio_duration * self.config.sample_rate)
        target_audio_len = target_audio_samples // self.config.hop_length + 1
        max_duration = np.array([ref_audio_len + target_audio_len], dtype=np.int64)

        combined_text = [list(reference_text + target_text)]
        text_ids = self.text_processor.text_to_indices(combined_text)
        time_step = np.array([0], dtype=np.int32)

        print(
            f"Single chunk: {len(target_text)} chars, total duration {total_estimated_duration:.1f}s. "
            f"Content: {target_text}"
        )
        return audio, text_ids, max_duration, time_step

    def _run_preprocess(
        self, audio: np.ndarray, text_ids: np.ndarray, max_duration: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Run preprocessing model"""
        session = self.model_session_manager.sessions["preprocess"]
        input_names = self.model_session_manager.input_names["preprocess"]
        output_names = self.model_session_manager.output_names["preprocess"]

        inputs = {
            input_names[0]: audio,
            input_names[1]: text_ids,
            input_names[2]: max_duration,
        }

        return session.run(output_names, inputs)

    def _run_transformer_steps(
        self,
        noise: np.ndarray,
        rope_cos_q: np.ndarray,
        rope_sin_q: np.ndarray,
        rope_cos_k: np.ndarray,
        rope_sin_k: np.ndarray,
        cat_mel_text: np.ndarray,
        cat_mel_text_drop: np.ndarray,
        time_step: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run transformer model iteratively"""
        session = self.model_session_manager.sessions["transformer"]
        input_names = self.model_session_manager.input_names["transformer"]
        output_names = self.model_session_manager.output_names["transformer"]

        for i in tqdm(
            range(0, self.config.nfe_step - 1, self.config.fuse_nfe),
            desc="Processing",
            total=self.config.nfe_step // self.config.fuse_nfe - 1,
        ):

            inputs = {
                input_names[0]: noise,
                input_names[1]: rope_cos_q,
                input_names[2]: rope_sin_q,
                input_names[3]: rope_cos_k,
                input_names[4]: rope_sin_k,
                input_names[5]: cat_mel_text,
                input_names[6]: cat_mel_text_drop,
                input_names[7]: time_step,
            }

            noise, time_step = session.run(output_names, inputs)

        return noise, time_step

    def _run_decode(self, noise: np.ndarray, ref_signal_len: np.ndarray) -> np.ndarray:
        """Run decode model to generate final audio"""
        session = self.model_session_manager.sessions["decode"]
        input_names = self.model_session_manager.input_names["decode"]
        output_names = self.model_session_manager.output_names["decode"]

        inputs = {input_names[0]: noise, input_names[1]: ref_signal_len}

        return session.run(output_names, inputs)[0]

    def synthesize(
        self,
        text: str,
        gender: Optional[str] = None,
        group: Optional[str] = None,
        area: Optional[str] = None,
        emotion: Optional[str] = None,
        output_path: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Synthesize speech from text

        Args:
            text: Target text to synthesize
            reference_audio: Path to reference audio file (optional, uses default if not provided)
            reference_text: Reference text matching the reference audio (optional, uses default if not provided)
            output_path: Path to save the generated audio (optional)

        Returns:
            Tuple of (generated_audio, generation_time)
        """
        start_time = time.time()

        ref_audio, ref_text = self.model_session_manager.select_sample(
            gender, group, area, emotion, reference_audio, reference_text
        )

        try:
            audio, text_ids, max_duration, time_step = self._prepare_inputs(
                ref_audio, ref_text, text
            )
            print("Generating speech for single chunk...")

            preprocess_outputs = self._run_preprocess(audio, text_ids, max_duration)
            (
                noise,
                rope_cos_q,
                rope_sin_q,
                rope_cos_k,
                rope_sin_k,
                cat_mel_text,
                cat_mel_text_drop,
                ref_signal_len,
            ) = preprocess_outputs

            noise, time_step = self._run_transformer_steps(
                noise,
                rope_cos_q,
                rope_sin_q,
                rope_cos_k,
                rope_sin_k,
                cat_mel_text,
                cat_mel_text_drop,
                time_step,
            )

            final_wave = self._run_decode(noise, ref_signal_len)

            generation_time = time.time() - start_time

            if output_path:
                self.audio_processor.save_audio(
                    final_wave, output_path, self.config.sample_rate
                )
                print(f"Audio saved to: {output_path}")

            return final_wave, generation_time

        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")

    def validate_configuration(self, reference_audio: Optional[str] = None) -> bool:
        """Validate configuration with reference audio"""
        if reference_audio is None:
            # If no reference audio is provided, configuration is valid
            # since the model will use built-in samples
            print("✅ Configuration valid: Using built-in voice samples")
            return True
        else:
            # Validate with the provided reference audio
            return self.config.validate_with_reference_audio(reference_audio)
