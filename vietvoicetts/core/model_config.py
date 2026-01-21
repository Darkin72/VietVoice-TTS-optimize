"""
Configuration management for TTS inference
"""

import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Model constants
MODEL_GENDER = ["male", "female"]
MODEL_GROUP = ["story", "news", "audiobook", "interview", "review"]
MODEL_AREA = ["northern", "southern", "central"]
MODEL_EMOTION = ["neutral", "serious", "monotone", "sad", "surprised", "happy", "angry"]


@dataclass
class ModelConfig:
    """Configuration for TTS model inference"""

    # Model settings
    model_url: str = (
        "https://huggingface.co/nguyenvulebinh/VietVoice-TTS/resolve/main/model-bin.pt"
    )
    model_cache_dir: str = "~/.cache/vietvoicetts"
    model_filename: str = "model-bin.pt"
    nfe_step: int = 32
    fuse_nfe: int = 1
    sample_rate: int = 24000
    speed: float = 1.0
    random_seed: int = 9527
    hop_length: int = 256

    # Text processing
    pause_punctuation: str = r".,?!:"

    # Audio processing
    cross_fade_duration: float = (
        0.3  # Duration in seconds for cross-fading between chunks
    )
    max_chunk_duration: float = 20.0  # Maximum duration in seconds for each chunk
    min_target_duration: float = 1.0  # Minimum duration in seconds for target audio

    # ONNX Runtime settings
    log_severity_level: int = 4
    log_verbosity_level: int = 4
    inter_op_num_threads: int = 0
    intra_op_num_threads: int = 0
    enable_cpu_mem_arena: bool = True

    # Optimization settings for H100
    use_fp16: bool = False
    use_io_binding: bool = False
    use_cuda_graph: bool = False
    streaming: bool = False

    # TTFB Optimization
    first_chunk_nfe_step: Optional[int] = (
        None  # If set, uses fewer steps for the first chunk
    )
    micro_chunking_words: int = (
        0  # Number of words for the first chunk to reduce TTFB (0 to disable)
    )

    def __post_init__(self):
        """Post-initialization validation"""
        self.validate_paths()

    @property
    def model_path(self) -> str:
        """Get the full path to the cached model file"""
        cache_dir = Path(self.model_cache_dir).expanduser()
        return str(cache_dir / self.model_filename)

    def ensure_model_downloaded(self) -> str:
        """Ensure model is downloaded and cached, return path to model file"""
        model_path = Path(self.model_path)
        cache_dir = model_path.parent

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model if it doesn't exist
        if not model_path.exists():
            print(f"Downloading model from {self.model_url}")
            print(f"Saving to {model_path}")

            try:
                # Download with progress indication
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, (block_num * block_size * 100) // total_size)
                        print(f"\rDownloading: {percent}%", end="", flush=True)

                urllib.request.urlretrieve(self.model_url, model_path, progress_hook)
                print(f"\n✅ Model downloaded successfully to {model_path}")

            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Failed to download model from {self.model_url}: {e}"
                )
            except Exception as e:
                # Clean up partial download
                if model_path.exists():
                    model_path.unlink()
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print(f"Using cached model: {model_path}")

        return str(model_path)

    def validate_paths(self):
        """Validate that required files exist or can be downloaded"""
        try:
            # Ensure model is available (download if needed)
            self.ensure_model_downloaded()
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}")

    def validate_with_reference_audio(self, reference_audio_path: str) -> bool:
        """Validate configuration against a reference audio file"""
        try:
            from pydub import AudioSegment

            audio_segment = (
                AudioSegment.from_file(reference_audio_path)
                .set_channels(1)
                .set_frame_rate(self.sample_rate)
            )
            ref_duration = len(audio_segment) / 1000.0  # Convert to seconds

            safety_margin = 1.0
            required_min_duration = (
                ref_duration + safety_margin + self.min_target_duration
            )

            if self.max_chunk_duration < required_min_duration:
                print(f"❌ Configuration Error:")
                print(f"   Reference audio: {ref_duration:.1f}s")
                print(f"   Min target duration: {self.min_target_duration:.1f}s")
                print(f"   Safety margin: {safety_margin:.1f}s")
                print(f"   Required max_chunk_duration: >{required_min_duration:.1f}s")
                print(f"   Current max_chunk_duration: {self.max_chunk_duration:.1f}s")
                return False
            else:
                print(f"✅ Configuration valid:")
                print(f"   Reference audio: {ref_duration:.1f}s")
                print(f"   Max chunk duration: {self.max_chunk_duration:.1f}s")
                print(
                    f"   Available target duration: {self.max_chunk_duration - ref_duration - safety_margin:.1f}s"
                )
                return True

        except Exception as e:
            print(f"❌ Error validating reference audio: {e}")
            return False

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# Backward compatibility alias
TTSConfig = ModelConfig
