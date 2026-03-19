"""
Text processing utilities for TTS inference
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict


class TextProcessor:
    """Handles text processing operations"""

    def __init__(self, vocab_path: str):
        self.vocab_char_map = self._load_vocab(vocab_path)
        self.vocab_size = len(self.vocab_char_map)

    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary mapping from file"""
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        vocab_char_map = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, char in enumerate(f):
                vocab_char_map[char.rstrip("\n")] = i
        return vocab_char_map

    def text_to_indices(self, texts: List[List[str]]) -> np.ndarray:
        """Convert text to indices using vocabulary mapping"""
        get_idx = self.vocab_char_map.get
        list_idx_tensors = [
            np.array([get_idx(c, 0) for c in text], dtype=np.int32) for text in texts
        ]
        return np.stack(list_idx_tensors, axis=0)

    def calculate_text_length(self, text: str, pause_punc: str) -> int:
        """Calculate text length including pause punctuation weighting"""
        return len(text.encode("utf-8")) + 3 * len(re.findall(pause_punc, text))

    def clean_text(self, text: str) -> str:
        """Clean text to keep only readable characters"""
        # only keep readable characters in alphabet, vietnamese characters, space, punctuation
        alphabet_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        vietnamese_chars = (
            "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳỵỷỹýỳỵỷỹ"
        )
        punctuation_chars = " .,!?'@$%&/:;()"
        all_valid_chars = (
            alphabet_chars
            + alphabet_chars.upper()
            + vietnamese_chars
            + vietnamese_chars.upper()
            + punctuation_chars
        )
        all_valid_chars = list(set(all_valid_chars))

        # replace all invalid characters with space using all_valid_chars and regex
        if "\n" in text:
            chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
            for idx, chunk in enumerate(chunks):
                if not chunk.strip().endswith("."):
                    chunks[idx] = chunk.strip() + "."
            text = " ".join(chunks)

        text = re.sub(f"[^{''.join(all_valid_chars)}]", " ", text)
        text = text.strip()
        # replace ;:() with ,
        text = re.sub(r"[;:()]", ",", text)

        # make sure no duplicate ,.
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r",+", ",", text)
        text = re.sub(r"\s+", " ", text)

        # Append . at the end of the text if it doesn't end with . or ? or ! or ,
        if not text.endswith((".", "?", "!", ",")):
            text += "."

        return text
