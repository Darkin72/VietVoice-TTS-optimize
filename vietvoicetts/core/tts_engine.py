"""
TTS Engine - Main speech synthesis engine
"""

import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Generator
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
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """Prepare all inputs for inference, handling text chunking if needed"""
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

        # Determine if chunking is needed
        # TTFB Optimization: Micro-chunking for the first chunk
        first_chunk = None
        remaining_text = target_text

        if self.config.micro_chunking_words > 0:
            words = target_text.split()
            if len(words) > self.config.micro_chunking_words:
                # Find a good split point (preferably at a punctuation)
                split_idx = self.config.micro_chunking_words
                # Look ahead a few words for punctuation
                for i in range(split_idx, min(split_idx + 5, len(words))):
                    if any(p in words[i - 1] for p in ".,!?:"):
                        split_idx = i
                        break

                first_chunk = " ".join(words[:split_idx])
                # Ensure it ends with proper punctuation for natural flow
                if not first_chunk.endswith((".", "?", "!", ",")):
                    first_chunk += ","
                remaining_text = " ".join(words[split_idx:])

        # Calculate max characters per chunk for remaining text
        safety_margin = 1.0
        available_target_duration = (
            self.config.max_chunk_duration - ref_audio_duration - safety_margin
        )

        if available_target_duration <= 0:
            raise ValueError(
                f"Reference audio duration ({ref_audio_duration:.1f}s) exceeds max chunk duration ({self.config.max_chunk_duration}s)"
            )

        max_chars_per_chunk = int(
            speaking_rate * available_target_duration * self.config.speed
        )

        chunks = []
        if first_chunk:
            chunks.append(first_chunk)

        if remaining_text:
            remaining_chunks = self.text_processor.chunk_text(
                remaining_text, max_chars=max_chars_per_chunk
            )
            chunks.extend(remaining_chunks)

        # Prepare inputs for each chunk
        inputs_list = []
        for i, chunk in enumerate(chunks):
            chunk_text_len = self.text_processor.calculate_text_length(
                chunk, self.config.pause_punctuation
            )

            # Calculate target duration
            chunk_target_duration = max(
                chunk_text_len / speaking_rate / self.config.speed,
                self.config.min_target_duration,
            )

            # Use lower nfe_step for first chunk if configured
            current_nfe = self.config.nfe_step
            if i == 0 and self.config.first_chunk_nfe_step:
                current_nfe = self.config.first_chunk_nfe_step

            target_audio_samples = int(chunk_target_duration * self.config.sample_rate)
            target_audio_len = target_audio_samples // self.config.hop_length + 1
            chunk_audio_len = ref_audio_len + target_audio_len

            max_duration = np.array([chunk_audio_len], dtype=np.int64)

            combined_text = [list(reference_text + chunk)]
            text_ids = self.text_processor.text_to_indices(combined_text)
            time_step = np.array([0], dtype=np.int32)

            inputs_list.append((audio, text_ids, max_duration, time_step, current_nfe))

            print(
                f"Chunk {i+1}/{len(chunks)} ({current_nfe} steps): {len(chunk)} chars. Content: {chunk[:50]}..."
            )

        return inputs_list

    def _run_preprocess(
        self, audio: np.ndarray, text_ids: np.ndarray, max_duration: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Run preprocessing model with automatic type conversion"""
        session = self.model_session_manager.sessions["preprocess"]
        input_names = self.model_session_manager.input_names["preprocess"]
        output_names = self.model_session_manager.output_names["preprocess"]

        # Handle type conversion matching the reference server logic.
        # Reference server passes int16 (range ~32k) to the model.
        input_info = session.get_inputs()[0]
        if "float" in input_info.type.lower():
            # If model wants float, provide float32 but maintain the magnitude (range ~32k)
            # as the reference server does not divide by 32768 in Python code.
            audio = audio.astype(np.float32)
        elif "int16" in input_info.type.lower() and audio.dtype != np.int16:
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

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
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run transformer model iteratively"""
        session = self.model_session_manager.sessions["transformer"]
        input_names = self.model_session_manager.input_names["transformer"]
        output_names = self.model_session_manager.output_names["transformer"]

        has_cuda = (
            "CUDAExecutionProvider" in session.get_providers()
            or "TensorrtExecutionProvider" in session.get_providers()
        )

        if self.config.use_io_binding and has_cuda:
            # IO Binding for faster inference
            io_binding = session.io_binding()
            device_id = 0

            try:
                import torch

                # Use torch for persistent GPU buffers
                noise_tensor = torch.from_numpy(noise).cuda(device_id)
                time_step_tensor = torch.from_numpy(time_step).cuda(device_id)

                # Bind static inputs once to avoid transfers
                def bind_gpu(name, arr):
                    t = torch.from_numpy(arr).cuda(device_id)
                    io_binding.bind_input(
                        name,
                        device_type="cuda",
                        device_id=device_id,
                        element_type=arr.dtype,
                        shape=t.shape,
                        buffer_ptr=t.data_ptr(),
                    )
                    return t

                _rcq = bind_gpu(input_names[1], rope_cos_q)
                _rsq = bind_gpu(input_names[2], rope_sin_q)
                _rck = bind_gpu(input_names[3], rope_cos_k)
                _rsk = bind_gpu(input_names[4], rope_sin_k)
                _cmt = bind_gpu(input_names[5], cat_mel_text)
                _cmtd = bind_gpu(input_names[6], cat_mel_text_drop)

                for _ in range(0, steps - 1, self.config.fuse_nfe):
                    io_binding.bind_input(
                        input_names[0],
                        device_type="cuda",
                        device_id=device_id,
                        element_type=np.float32,
                        shape=noise_tensor.shape,
                        buffer_ptr=noise_tensor.data_ptr(),
                    )
                    io_binding.bind_input(
                        input_names[7],
                        device_type="cuda",
                        device_id=device_id,
                        element_type=np.int32,
                        shape=time_step_tensor.shape,
                        buffer_ptr=time_step_tensor.data_ptr(),
                    )

                    new_noise = torch.empty_like(noise_tensor)
                    new_time_step = torch.empty_like(time_step_tensor)

                    io_binding.bind_output(
                        output_names[0],
                        device_type="cuda",
                        device_id=device_id,
                        element_type=np.float32,
                        shape=new_noise.shape,
                        buffer_ptr=new_noise.data_ptr(),
                    )
                    io_binding.bind_output(
                        output_names[1],
                        device_type="cuda",
                        device_id=device_id,
                        element_type=np.int32,
                        shape=new_time_step.shape,
                        buffer_ptr=new_time_step.data_ptr(),
                    )

                    session.run_with_iobinding(io_binding)

                    noise_tensor = new_noise
                    time_step_tensor = new_time_step

                return noise_tensor.cpu().numpy(), time_step_tensor.cpu().numpy()
            except ImportError:
                print(
                    "Warning: torch not found, falling back to standard ORT inference for IO binding"
                )
                # Fallback to standard loop if torch isn't available for buffer management

        # Default path
        for _ in range(0, steps - 1, self.config.fuse_nfe):
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

    def synthesize_stream(
        self,
        text: str,
        gender: Optional[str] = None,
        group: Optional[str] = None,
        area: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech from text and stream audio chunks with cross-fading for smooth transitions.
        """
        ref_audio, ref_text = self.model_session_manager.select_sample(
            gender, group, area, emotion, reference_audio, reference_text
        )

        inputs_list = self._prepare_inputs(ref_audio, ref_text, text)

        cross_fade_duration = self.config.cross_fade_duration
        sample_rate = self.config.sample_rate
        cross_fade_samples = int(cross_fade_duration * sample_rate)
        prev_chunk_tail = None

        for i, (audio, text_ids, max_duration, time_step, steps) in enumerate(
            inputs_list
        ):
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
                steps,
            )

            current_chunk = self._run_decode(noise, ref_signal_len).squeeze()
            current_chunk = self.audio_processor.fix_clipped_audio(current_chunk)

            # Streaming logic with cross-fade (Adapted from reference implementation)
            if len(inputs_list) == 1:
                yield current_chunk
                return

            if prev_chunk_tail is None:
                if len(current_chunk) > cross_fade_samples:
                    to_yield = current_chunk[:-cross_fade_samples]
                    prev_chunk_tail = current_chunk[-cross_fade_samples:]
                    yield to_yield
                else:
                    prev_chunk_tail = current_chunk
            else:
                actual_cross_fade = min(
                    len(prev_chunk_tail), len(current_chunk), cross_fade_samples
                )
                if actual_cross_fade > 0:
                    prev_overlap = prev_chunk_tail[-actual_cross_fade:]
                    next_overlap = current_chunk[:actual_cross_fade]

                    # Smooth transition
                    fade_out = np.cos(np.linspace(0, np.pi / 2, actual_cross_fade)) ** 2
                    fade_in = np.sin(np.linspace(0, np.pi / 2, actual_cross_fade)) ** 2

                    cross_faded = (
                        prev_overlap.astype(np.float32) * fade_out
                        + next_overlap.astype(np.float32) * fade_in
                    )

                    # For consistency with input chunk type
                    if current_chunk.dtype == np.int16:
                        cross_faded = cross_faded.astype(np.int16)

                    if len(prev_chunk_tail) > actual_cross_fade:
                        yield prev_chunk_tail[:-actual_cross_fade]
                    yield cross_faded

                    if i == len(inputs_list) - 1:
                        yield current_chunk[actual_cross_fade:]
                    else:
                        if len(current_chunk) > actual_cross_fade + cross_fade_samples:
                            yield current_chunk[actual_cross_fade:-cross_fade_samples]
                            prev_chunk_tail = current_chunk[-cross_fade_samples:]
                        else:
                            prev_chunk_tail = current_chunk[actual_cross_fade:]
                else:
                    yield prev_chunk_tail
                    if i == len(inputs_list) - 1:
                        yield current_chunk
                    else:
                        prev_chunk_tail = current_chunk

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

        try:
            generated_waves = []
            for chunk in self.synthesize_stream(
                text, gender, group, area, emotion, reference_audio, reference_text
            ):
                generated_waves.append(chunk)

            # Concatenate all generated waves
            if len(generated_waves) > 1:
                # Chunks from synthesize_stream are already cross-faded
                final_wave = np.concatenate([w.reshape(-1) for w in generated_waves])
            elif len(generated_waves) == 1:
                final_wave = generated_waves[0].reshape(-1)
            else:
                final_wave = np.array([])

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
