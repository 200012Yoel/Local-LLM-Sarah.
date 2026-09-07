"""
Tokenizer Byte-Pair Encoding (BPE) Multilingue conçu de zéro pour Sarah Ngin.
Garantit 100% de couverture de vocabulaire (Français, Hébreu, Chinois, Anglais)
sans risque de token inconnu (Unicode byte-level fallback).
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SarahTokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self.pad_token_id = self.special_tokens["<pad>"]
        self.bos_token_id = self.special_tokens["<bos>"]
        self.eos_token_id = self.special_tokens["<eos>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        
        # Initialisation du vocabulaire avec les 256 octets standards (Byte-level)
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
        """
        Entraîne l'algorithme BPE ultra-rapidement en agrégeant les fréquences de mots.
        """
        self._init_base_vocab()
        self.merges = []

        offset = len(self.special_tokens)
        
        # 1. Extraction et comptage des mots/segments fréquents
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            for word in text.split(" "):
                if word:
                    word_counts[" " + word] += 1

        # 2. Conversion en tuples d'octets (sur les 4000 mots les plus fréquents)
        vocab_words = {}
        for word, freq in word_counts.most_common(4000):
            raw_bytes = word.encode("utf-8")
            vocab_words[tuple(offset + b for b in raw_bytes)] = freq

        num_merges = (self.target_vocab_size - len(self.vocab)) if max_merges is None else max_merges
        logger.info(f"Entraînement rapide du Tokenizer Sarah Ngin : {num_merges} fusions ciblées sur {len(vocab_words)} mots uniques...")

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

            # Mettre à jour les mots du vocabulaire
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

        logger.info(f"Tokenizer entraîné avec succès en mode rapide. Taille du vocabulaire : {self.vocab_size}")

    def _encode_single_word(self, word: str) -> List[int]:
        """Encode un mot ou segment unique avec application des merges."""
        if not hasattr(self, "_encode_cache"):
            self._encode_cache = {}
        if word in self._encode_cache:
            return self._encode_cache[word]

        raw_bytes = word.encode("utf-8")
        offset = len(self.special_tokens)
        ids = [offset + b for b in raw_bytes]

        for p0_bytes, p1_bytes in self.merges:
            if len(ids) < 2:
                break
            p0_id = self.inv_vocab.get(p0_bytes)
            p1_id = self.inv_vocab.get(p1_bytes)
            if p0_id is None or p1_id is None:
                continue
            merged_id = self.inv_vocab.get(p0_bytes + p1_bytes)
            if merged_id is None:
                continue

            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0_id and ids[i+1] == p1_id:
                    new_ids.append(merged_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        self._encode_cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode une chaîne de caractères en IDs de tokens ultra-rapidement avec cache.
        """
        if not text:
            tokens = []
            if add_bos: tokens = [self.bos_token_id] + tokens
            if add_eos: tokens = tokens + [self.eos_token_id]
            return tokens

        tokens = []
        words = text.split(" ")
        for idx, word in enumerate(words):
            segment = (" " if idx > 0 else "") + word
            tokens.extend(self._encode_single_word(segment))

        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Décode une liste de tokens en texte UTF-8.
        """
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
        # Décodage avec remplacement tolérant si un octet isolé
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Sauvegarde le vocabulaire et les règles de merge au format JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "special_tokens": self.special_tokens,
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [(list(p0), list(p1)) for p0, p1 in self.merges]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Tokenizer sauvegardé dans {path}")

    @classmethod
    def load(cls, filepath: str) -> "SarahTokenizer":
        """Recharge un tokenizer sauvegardé."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.pad_token_id = tokenizer.special_tokens["<pad>"]
        tokenizer.bos_token_id = tokenizer.special_tokens["<bos>"]
        tokenizer.eos_token_id = tokenizer.special_tokens["<eos>"]
        tokenizer.unk_token_id = tokenizer.special_tokens["<unk>"]

        tokenizer.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [(bytes(p0), bytes(p1)) for p0, p1 in data["merges"]]

        logger.info(f"Tokenizer chargé depuis {path} (Vocabulaire: {tokenizer.vocab_size})")
        return tokenizer
