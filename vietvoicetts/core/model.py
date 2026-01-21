"""
Model session management for ONNX Runtime
"""

import tarfile
import tempfile
import shutil
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json
import onnxruntime
import random

from .model_config import (
    ModelConfig,
    MODEL_GENDER,
    MODEL_GROUP,
    MODEL_AREA,
    MODEL_EMOTION,
)


class ModelSessionManager:
    """Manages ONNX Runtime sessions"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.providers = self._get_optimal_providers()
        self.sessions = {}
        self.input_names = {}
        self.output_names = {}
        self.sample_metadata = {}
        self.temp_dir = None
        self.vocab_path = None
        self.cached_ref_audio = None
        self.cached_ref_text = None

    def _get_optimal_providers(self) -> List[str]:
        """Get the fastest available providers"""
        available_providers = onnxruntime.get_available_providers()
        print(f"🔍 System detected providers: {available_providers}")

        provider_priority = []

        if self.config.use_tensorrt:
            provider_priority.append("TensorrtExecutionProvider")

        provider_priority.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])

        selected_providers = []
        for provider in provider_priority:
            if provider in available_providers:
                if provider == "TensorrtExecutionProvider":
                    trt_options = {
                        "trt_fp16_enable": self.config.use_fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(
                            Path(self.config.model_cache_dir).expanduser() / "trt_cache"
                        ),
                    }
                    if self.config.use_cuda_graph:
                        trt_options["trt_cuda_graph_enable"] = True
                    selected_providers.append((provider, trt_options))
                elif provider == "CUDAExecutionProvider":
                    cuda_options = {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "do_copy_in_default_stream": True,
                    }
                    selected_providers.append((provider, cuda_options))
                else:
                    selected_providers.append(provider)

        return selected_providers

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
        model_path = self.config.ensure_model_downloaded()
        expected_models = {
            "preprocess": "preprocess.onnx",
            "transformer": "transformer.onnx",
            "decode": "decode.onnx",
        }

        # Prepare samples directory
        samples_dir = Path(self.config.model_cache_dir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(model_path, "r") as tar:
                tar_members = tar.getnames()
                self.sample_metadata = json.load(tar.extractfile("audio_metadata.json"))

                # Extract audio samples
                print("📦 Checking/Extracting voice samples...")
                HARDCODED_FILE = "soctrangtv_7834_a1ad56dd.wav"  # <-- TÊN FILE

                for member in tar_members:
                    # Chỉ giải nén đúng file này
                    if member == HARDCODED_FILE:
                        filename = os.path.basename(member)
                        target_path = samples_dir / filename
                        if not target_path.exists():
                            source = tar.extractfile(member)
                            with open(target_path, "wb") as f:
                                f.write(source.read())

                # LOAD DIRECTLY TO RAM
                print(f"🚀 Pre-loading sample {HARDCODED_FILE} to RAM...")

                # Find metadata
                sample_meta = next(
                    (
                        s
                        for s in self.sample_metadata
                        if s["file_name"] == HARDCODED_FILE
                    ),
                    None,
                )
                if not sample_meta:
                    # Fallback to first
                    sample_meta = self.sample_metadata[0]
                    HARDCODED_FILE = sample_meta["file_name"]
                    print(
                        f"⚠️ Metadata not found for hardcoded file, falling back to: {HARDCODED_FILE}"
                    )

                # Read audio bytes
                target_path = samples_dir / HARDCODED_FILE
                if target_path.exists():
                    with open(target_path, "rb") as f:
                        self.cached_ref_audio = f.read()
                else:
                    # Try tar fallback if not extracted successfully for some reason
                    extracted_file = tar.extractfile("cleaned_audios/" + HARDCODED_FILE)
                    if extracted_file:
                        self.cached_ref_audio = extracted_file.read()
                    else:
                        raise FileNotFoundError(
                            f"Audio file '{HARDCODED_FILE}' not found in tar archive"
                        )

                self.cached_ref_text = sample_meta["text"]
                print(f"✅ Sample loaded to RAM. Text: {self.cached_ref_text[:30]}...")

                # Lấy danh sách providers tối ưu một lần duy nhất
                preferred_providers = self._get_optimal_providers()

                for model_name, filename in expected_models.items():
                    matching_member = next(
                        (m for m in tar_members if m.endswith(filename)), None
                    )
                    if not matching_member:
                        raise FileNotFoundError(f"Model file '{filename}' not found")

                    extracted_file = tar.extractfile(matching_member)
                    model_bytes = extracted_file.read()
                    session_opts = self._create_session_options()

                    session = None

                    # Thử Tầng 1: TensorRT (Thử riêng cho từng model)
                    if any(
                        "Tensorrt" in (p[0] if isinstance(p, tuple) else p)
                        for p in preferred_providers
                    ):
                        try:
                            # Chỉ lấy Tensorrt provider để thử
                            trt_p = [
                                p
                                for p in preferred_providers
                                if (p[0] if isinstance(p, tuple) else p)
                                == "TensorrtExecutionProvider"
                            ]
                            session = onnxruntime.InferenceSession(
                                model_bytes, sess_options=session_opts, providers=trt_p
                            )
                            print(f"🚀 {model_name}: TensorRT Acceleration ENABLED")
                        except Exception:
                            # Nếu fail (như lỗi INT16), im lặng lùi về CUDA
                            pass

                    # Thử Tầng 2: CUDA (Nếu TRT fail hoặc không có trong list)
                    if session is None:
                        try:
                            cuda_p = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                            session = onnxruntime.InferenceSession(
                                model_bytes, sess_options=session_opts, providers=cuda_p
                            )
                            actual = session.get_providers()
                            if "CUDAExecutionProvider" in actual:
                                print(f"🚀 {model_name}: CUDA Acceleration ENABLED")
                            else:
                                print(f"ℹ️ {model_name}: CPU Fallback (CUDA not picked)")
                        except Exception:
                            # Thử nốt CPU
                            session = onnxruntime.InferenceSession(
                                model_bytes,
                                sess_options=session_opts,
                                providers=["CPUExecutionProvider"],
                            )
                            print(f"ℹ️ {model_name}: Running on CPU")

                    self.sessions[model_name] = session
                    self.input_names[model_name] = [
                        inp.name for inp in session.get_inputs()
                    ]
                    self.output_names[model_name] = [
                        out.name for out in session.get_outputs()
                    ]

                # In tổng kết
                final_status = {
                    k: v.get_providers()[0] for k, v in self.sessions.items()
                }
                print(f"✅ System Ready. Model Status: {final_status}")

                vocab_member = next(
                    (m for m in tar_members if m.endswith("vocab.txt")), None
                )
                if not vocab_member:
                    raise FileNotFoundError("Vocabulary file 'vocab.txt' not found")

                self.temp_dir = tempfile.mkdtemp(prefix="tts_vocab_")
                vocab_temp_path = Path(self.temp_dir) / "vocab.txt"
                with open(vocab_temp_path, "wb") as f:
                    f.write(tar.extractfile(vocab_member).read())
                self.vocab_path = str(vocab_temp_path)

        except Exception as e:
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def load_models(self) -> None:
        """Load all ONNX models from downloaded model file"""
        onnxruntime.set_seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        self._load_models_from_file()

    def select_sample(
        self,
        gender: Optional[str] = None,
        group: Optional[str] = None,
        area: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Select a sample from the metadata"""
        # Optimized path: Use RAM cached sample if no custom voice is requested
        if all(
            v is None
            for v in [gender, group, area, emotion, reference_audio, reference_text]
        ):
            if self.cached_ref_audio and self.cached_ref_text:
                return self.cached_ref_audio, self.cached_ref_text

        filter_options = {}
        if gender is not None:
            if gender not in MODEL_GENDER:
                raise ValueError(
                    f"Invalid gender: {gender}. Must be one of {MODEL_GENDER}"
                )
            filter_options["gender"] = gender
        if group is not None:
            if group not in MODEL_GROUP:
                raise ValueError(
                    f"Invalid group: {group}. Must be one of {MODEL_GROUP}"
                )
            filter_options["group"] = group
        if area is not None:
            if area not in MODEL_AREA:
                raise ValueError(f"Invalid area: {area}. Must be one of {MODEL_AREA}")
            filter_options["area"] = area
        if emotion is not None:
            if emotion not in MODEL_EMOTION:
                raise ValueError(
                    f"Invalid emotion: {emotion}. Must be one of {MODEL_EMOTION}"
                )
            filter_options["emotion"] = emotion

        if reference_audio is not None:
            if reference_text is None:
                raise ValueError(
                    "Reference text is required when using reference audio"
                )
            if not Path(reference_audio).exists():
                raise FileNotFoundError(
                    f"Reference audio file not found: {reference_audio}"
                )
            if len(filter_options) > 0:
                raise ValueError(
                    f"Cannot use reference audio and text with options: {list(filter_options.keys())}"
                )
            print(f"Using reference audio and text: {reference_audio}")
            return reference_audio, reference_text

        try:
            available_samples = []

            if len(filter_options) == 0:
                available_samples = [
                    (sample, idx) for idx, sample in enumerate(self.sample_metadata)
                ]
            else:
                for idx, sample in enumerate(self.sample_metadata):
                    if all(
                        sample[key] == value for key, value in filter_options.items()
                    ):
                        available_samples.append((sample, idx))

            if len(available_samples) == 0:
                sample, sample_idx = self.sample_metadata[0], 0
            else:
                sample, sample_idx = random.choice(available_samples)

            print(
                f"Selected sample #{sample_idx} with gender: {sample['gender']}, group: {sample['group']}, area: {sample['area']}, emotion: {sample['emotion']}"
            )

            # Get the cached model path
            model_path = self.config.ensure_model_downloaded()

            with tarfile.open(model_path, "r") as tar:
                ref_audio = tar.extractfile("cleaned_audios/" + sample["file_name"])
                if not ref_audio:
                    raise FileNotFoundError(
                        f"Audio file {sample['file_name']} not found in model archive"
                    )
                ref_audio = ref_audio.read()
                ref_text = sample["text"]
        except KeyError:
            raise ValueError(
                f"Sample not found for gender: {gender}, group: {group}, area: {area}, emotion: {emotion}"
            )
        return ref_audio, ref_text

    def cleanup(self) -> None:
        """Clean up temporary files and cached data"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.vocab_path = None
        self.cached_ref_audio = None
        self.cached_ref_text = None

    def __del__(self):
        self.cleanup()
