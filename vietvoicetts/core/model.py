"""
Model session management for ONNX Runtime
"""

import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json
import onnxruntime

from .model_config import ModelConfig


HARDCODED_REF_AUDIO_FILENAME = "soctrangtv_7834_a1ad56dd.wav"


class ModelSessionManager:
    """Manages ONNX Runtime sessions"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.providers = self._get_optimal_providers()
        self.sessions = {}
        self.input_names = {}
        self.output_names = {}
        self.sample_metadata = {}
        self.sample_audio_cache = {}
        self.temp_dir = None
        self.vocab_path = None

    def _get_optimal_providers(self) -> List[str | tuple[str, dict[str, str]]]:
        """Get the fastest available providers"""
        available_providers = onnxruntime.get_available_providers()

        provider_priority = [
            "TensorRTExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        selected_providers: List[str | tuple[str, dict[str, str]]] = []
        for provider in provider_priority:
            if (
                provider == "TensorRTExecutionProvider"
                and provider in available_providers
                and self.config.enable_tensorrt
            ):
                trt_options = {
                    "device_id": str(self.config.cuda_device_id),
                    "trt_fp16_enable": "1" if self.config.trt_fp16_enable else "0",
                    "trt_engine_cache_enable": (
                        "1" if self.config.trt_engine_cache_enable else "0"
                    ),
                    "trt_engine_cache_path": str(
                        Path(self.config.trt_engine_cache_path).expanduser()
                    ),
                    "trt_max_workspace_size": str(self.config.trt_max_workspace_size),
                }
                selected_providers.append(("TensorRTExecutionProvider", trt_options))
            elif (
                provider == "CUDAExecutionProvider" and provider in available_providers
            ):
                cuda_options = {
                    "device_id": str(self.config.cuda_device_id),
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": self.config.cuda_conv_algo_search,
                    "do_copy_in_default_stream": (
                        "1" if self.config.cuda_copy_in_default_stream else "0"
                    ),
                    "cudnn_conv_use_max_workspace": (
                        "1" if self.config.cuda_conv_use_max_workspace else "0"
                    ),
                }
                if self.config.enable_cuda_graph:
                    cuda_options["enable_cuda_graph"] = "1"
                selected_providers.append(("CUDAExecutionProvider", cuda_options))
            elif provider == "CPUExecutionProvider" and provider in available_providers:
                selected_providers.append("CPUExecutionProvider")

        has_cpu_provider = any(
            p == "CPUExecutionProvider"
            or (isinstance(p, tuple) and p[0] == "CPUExecutionProvider")
            for p in selected_providers
        )
        if not has_cpu_provider:
            selected_providers.append("CPUExecutionProvider")

        return selected_providers

    @staticmethod
    def _providers_without_tensorrt(
        providers: List[str | tuple[str, dict[str, str]]],
    ) -> List[str | tuple[str, dict[str, str]]]:
        return [
            p
            for p in providers
            if not (isinstance(p, tuple) and p[0] == "TensorRTExecutionProvider")
            and p != "TensorRTExecutionProvider"
        ]

    def _create_session_options(self) -> onnxruntime.SessionOptions:
        """Create optimized ONNX Runtime session options"""
        session_opts = onnxruntime.SessionOptions()
        session_opts.log_severity_level = self.config.log_severity_level
        session_opts.log_verbosity_level = self.config.log_verbosity_level
        session_opts.inter_op_num_threads = self.config.inter_op_num_threads
        session_opts.intra_op_num_threads = self.config.intra_op_num_threads
        session_opts.enable_cpu_mem_arena = self.config.enable_cpu_mem_arena
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        return session_opts

    def _load_models_from_file(self) -> None:
        """Load ONNX models from downloaded model file and extract vocab"""
        # Ensure model is downloaded and get path
        model_path = self.config.ensure_model_downloaded()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        expected_models = {
            "preprocess": "preprocess.onnx",
            "transformer": "transformer.onnx",
            "decode": "decode.onnx",
        }

        try:
            with tarfile.open(model_path, "r") as tar:
                tar_members = tar.getnames()

                # Load metadata.json
                metadata_file = tar.extractfile("audio_metadata.json")
                if not metadata_file:
                    raise FileNotFoundError(
                        "audio_metadata.json not found in model archive"
                    )
                self.sample_metadata = json.load(metadata_file)

                # Load ONNX models
                active_providers = self.providers
                for model_name, filename in expected_models.items():
                    matching_member = next(
                        (m for m in tar_members if m.endswith(filename)), None
                    )
                    if not matching_member:
                        raise FileNotFoundError(
                            f"Model file '{filename}' not found in model archive"
                        )

                    extracted_file = tar.extractfile(matching_member)
                    if not extracted_file:
                        raise RuntimeError(
                            f"Failed to extract {filename} from model archive"
                        )

                    model_bytes = extracted_file.read()
                    session_opts = self._create_session_options()
                    try:
                        session = onnxruntime.InferenceSession(
                            model_bytes,
                            sess_options=session_opts,
                            providers=active_providers,
                        )
                    except Exception as session_error:
                        can_retry_without_trt = any(
                            (
                                isinstance(p, tuple)
                                and p[0] == "TensorRTExecutionProvider"
                            )
                            or p == "TensorRTExecutionProvider"
                            for p in active_providers
                        )
                        if not can_retry_without_trt:
                            raise

                        active_providers = self._providers_without_tensorrt(
                            active_providers
                        )
                        print(
                            "Warning: TensorRT session init failed. "
                            "Retrying with CUDAExecutionProvider. "
                            f"Details: {session_error}"
                        )
                        session = onnxruntime.InferenceSession(
                            model_bytes,
                            sess_options=session_opts,
                            providers=active_providers,
                        )

                    self.sessions[model_name] = session
                    self.input_names[model_name] = [
                        inp.name for inp in session.get_inputs()
                    ]
                    self.output_names[model_name] = [
                        out.name for out in session.get_outputs()
                    ]

                self.providers = active_providers

                for sample in self.sample_metadata:
                    file_name = sample.get("file_name")
                    if not file_name:
                        continue
                    member_name = "cleaned_audios/" + file_name
                    extracted_audio = tar.extractfile(member_name)
                    if not extracted_audio:
                        raise FileNotFoundError(
                            f"Audio file {file_name} not found in model archive"
                        )
                    self.sample_audio_cache[file_name] = extracted_audio.read()

                # Extract vocab.txt to temporary file
                vocab_member = next(
                    (m for m in tar_members if m.endswith("vocab.txt")), None
                )
                if not vocab_member:
                    raise FileNotFoundError(
                        "Vocabulary file 'vocab.txt' not found in model archive"
                    )

                self.temp_dir = tempfile.mkdtemp(prefix="tts_vocab_")
                vocab_temp_path = Path(self.temp_dir) / "vocab.txt"

                extracted_file = tar.extractfile(vocab_member)
                if not extracted_file:
                    raise RuntimeError("Failed to extract vocab.txt from model archive")

                with open(vocab_temp_path, "wb") as f:
                    f.write(extracted_file.read())

                self.vocab_path = str(vocab_temp_path)

        except Exception as e:
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            raise RuntimeError(f"Failed to load models from file: {str(e)}")

    def load_models(self) -> None:
        """Load all ONNX models from downloaded model file"""
        onnxruntime.set_seed(self.config.random_seed)
        self._load_models_from_file()

    def select_sample(
        self,
        gender: Optional[str] = None,
        group: Optional[str] = None,
        area: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[str | bytes, str]:
        """Select a hardcoded built-in sample unless custom reference audio is provided."""
        if reference_audio is not None:
            if reference_text is None:
                raise ValueError(
                    "Reference text is required when using reference audio"
                )
            if not Path(reference_audio).exists():
                raise FileNotFoundError(
                    f"Reference audio file not found: {reference_audio}"
                )
            return reference_audio, reference_text

        sample = next(
            (
                item
                for item in self.sample_metadata
                if item.get("file_name") == HARDCODED_REF_AUDIO_FILENAME
            ),
            None,
        )
        if sample is None:
            raise ValueError(
                f"Hardcoded sample '{HARDCODED_REF_AUDIO_FILENAME}' not found in model metadata"
            )

        file_name = sample["file_name"]
        if file_name in self.sample_audio_cache:
            ref_audio = self.sample_audio_cache[file_name]
        else:
            model_path = self.config.ensure_model_downloaded()
            with tarfile.open(model_path, "r") as tar:
                extracted_audio = tar.extractfile("cleaned_audios/" + file_name)
                if not extracted_audio:
                    raise FileNotFoundError(
                        f"Audio file {file_name} not found in model archive"
                    )
                ref_audio = extracted_audio.read()
        ref_text = sample["text"]
        return ref_audio, ref_text

    def cleanup(self) -> None:
        """Clean up temporary files"""
        self.sample_audio_cache.clear()
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.vocab_path = None

    def __del__(self):
        self.cleanup()
