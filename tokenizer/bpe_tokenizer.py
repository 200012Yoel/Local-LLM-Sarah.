"""
Tokenizer Byte-Pair Encoding (BPE) Multilingue conçu de zéro pour Sarah Ngin.
Garantit 100% de couverture de vocabulaire (Français, Anglais, Hébreu, Code)
avec support natif des tokens de rôles conversationnels (<user>, <assistant>, <system>).
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SarahTokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.target_vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<user>": 4,
            "<assistant>": 5,
            "<system>": 6,
        }
        self.pad_token_id = self.special_tokens["<pad>"]
        self.bos_token_id = self.special_tokens["<bos>"]
        self.eos_token_id = self.special_tokens["<eos>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        self.user_token_id = self.special_tokens["<user>"]
        self.assistant_token_id = self.special_tokens["<assistant>"]
        self.system_token_id = self.special_tokens["<system>"]

        self.vocab: Dict[int, bytes] = {}
        self.inv_vocab: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self._encode_cache: Dict[str, List[int]] = {}

        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialise les tokens spéciaux et les 256 octets bruts pour zéro OOV."""
        self.vocab.clear()
        self.inv_vocab.clear()

        # Special tokens
        for name, idx in self.special_tokens.items():
            b_name = name.encode("utf-8")
            self.vocab[idx] = b_name
            self.inv_vocab[b_name] = idx

        # Base 256 bytes
        offset = len(self.special_tokens)
        for b in range(256):
            byte_val = bytes([b])
            idx = offset + b
            self.vocab[idx] = byte_val
            self.inv_vocab[byte_val] = idx

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, texts: List[str], max_merges: Optional[int] = None):
        """Entraîne l'algorithme BPE en agrégeant les fréquences de mots."""
        self._init_base_vocab()
        self.merges = []

        offset = len(self.special_tokens)

        from collections import Counter
        word_counts = Counter()
        for text in texts:
            # Ne pas découper les tokens spéciaux
            clean_text = text
            for st in self.special_tokens.keys():
                clean_text = clean_text.replace(st, " ")
            for word in clean_text.split(" "):
                if word:
                    word_counts[" " + word] += 1

        vocab_words = {}
        for word, freq in word_counts.most_common(6000):
            raw_bytes = word.encode("utf-8")
            vocab_words[tuple(offset + b for b in raw_bytes)] = freq

        num_merges = (self.target_vocab_size - len(self.vocab)) if max_merges is None else max_merges
        logger.info(f"Entraînement BPE : {num_merges} fusions ciblées...")

        for i in range(num_merges):
            pairs = {}
            for word_tuple, freq in vocab_words.items():
                for p0, p1 in zip(word_tuple, word_tuple[1:]):
                    pair = (p0, p1)
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            new_idx = len(self.vocab)
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_idx] = merged_bytes
            self.inv_vocab[merged_bytes] = new_idx
            self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

            new_vocab_words = {}
            p0, p1 = best_pair
            for word_tuple, freq in vocab_words.items():
                new_word = []
                j = 0
                while j < len(word_tuple):
                    if j < len(word_tuple) - 1 and word_tuple[j] == p0 and word_tuple[j+1] == p1:
                        new_word.append(new_idx)
                        j += 2
                    else:
                        new_word.append(word_tuple[j])
                        j += 1
                new_vocab_words[tuple(new_word)] = freq
            vocab_words = new_vocab_words

    def _ensure_bpe_ranks(self):
        if not hasattr(self, "_bpe_ranks") or len(self._bpe_ranks) != len(self.merges):
            self._bpe_ranks = {(p0, p1): idx for idx, (p0, p1) in enumerate(self.merges)}

    def _encode_single_word(self, word: str) -> List[int]:
        if not hasattr(self, "_encode_cache"):
            self._encode_cache = {}
        if word in self._encode_cache:
            return self._encode_cache[word]

        self._ensure_bpe_ranks()
        raw_bytes = word.encode("utf-8")
        offset = len(self.special_tokens)
        parts = [self.vocab[offset + b] for b in raw_bytes]

        while len(parts) >= 2:
            pairs = [(parts[i], parts[i+1]) for i in range(len(parts)-1)]
            ranks = [(self._bpe_ranks.get(p, float('inf')), i, p) for i, p in enumerate(pairs) if p in self._bpe_ranks]
            if not ranks:
                break
            _, min_i, best_pair = min(ranks)
            merged = best_pair[0] + best_pair[1]
            parts = parts[:min_i] + [merged] + parts[min_i+2:]

        ids = [self.inv_vocab[p] for p in parts if p in self.inv_vocab]
        self._encode_cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode une chaîne de caractères en tokens en préservant les balises de rôles spéciales."""
        if not text:
            tokens = []
            if add_bos: tokens = [self.bos_token_id] + tokens
            if add_eos: tokens = tokens + [self.eos_token_id]
            return tokens

        special_pattern = r"(<pad>|<bos>|<eos>|<unk>|<user>|<assistant>|<system>)"
        parts = re.split(special_pattern, text)

        tokens = []
        if add_bos and not text.startswith("<bos>"):
            tokens.append(self.bos_token_id)

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                words = part.split(" ")
                for idx, word in enumerate(words):
                    segment = (" " if idx > 0 else "") + word
                    tokens.extend(self._encode_single_word(segment))

        if add_eos and not text.endswith("<eos>"):
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Décode les tokens en chaîne de caractères UTF-8."""
        byte_chunks = []
        special_ids = set(self.special_tokens.values())

        for token in tokens:
            if skip_special_tokens and token in special_ids:
                continue
            if token in self.vocab:
                byte_chunks.append(self.vocab[token])
            else:
                byte_chunks.append(b"?")

        all_bytes = b"".join(byte_chunks)
        text = all_bytes.decode("utf-8", errors="replace").replace("\ufffd", "")
        return text

    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [(list(p0), list(p1)) for p0, p1 in self.merges]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "SarahTokenizer":
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.pad_token_id = tokenizer.special_tokens["<pad>"]
        tokenizer.bos_token_id = tokenizer.special_tokens["<bos>"]
        tokenizer.eos_token_id = tokenizer.special_tokens["<eos>"]
        tokenizer.unk_token_id = tokenizer.special_tokens["<unk>"]
        tokenizer.user_token_id = tokenizer.special_tokens.get("<user>", 4)
        tokenizer.assistant_token_id = tokenizer.special_tokens.get("<assistant>", 5)
        tokenizer.system_token_id = tokenizer.special_tokens.get("<system>", 6)

        tokenizer.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [(bytes(p0), bytes(p1)) for p0, p1 in data["merges"]]
        return tokenizer
