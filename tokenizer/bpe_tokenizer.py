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

    def _get_stats(self, ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Compte les paires adjacentes les plus fréquentes."""
        counts: Dict[Tuple[int, int], int] = {}
        for row in ids_list:
            for pair in zip(row, row[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids_list: List[List[int]], pair: Tuple[int, int], idx: int) -> List[List[int]]:
        """Remplace toutes les occurrences d'une paire par le nouveau token_id."""
        new_ids_list = []
        p0, p1 = pair
        for row in ids_list:
            new_row = []
            i = 0
            while i < len(row):
                if i < len(row) - 1 and row[i] == p0 and row[i + 1] == p1:
                    new_row.append(idx)
                    i += 2
                else:
                    new_row.append(row[i])
                    i += 1
            new_ids_list.append(new_row)
        return new_ids_list

    def train(self, texts: List[str], max_merges: Optional[int] = None):
        """
        Entraîne l'algorithme BPE sur la liste de textes.
        """
        self._init_base_vocab()
        self.merges = []
        
        # Convertit chaque texte en liste d'octets de base
        offset = len(self.special_tokens)
        ids_list: List[List[int]] = []
        for text in texts:
            if not text:
                continue
            raw_bytes = text.encode("utf-8")
            ids_list.append([offset + b for b in raw_bytes])

        num_merges = (self.target_vocab_size - len(self.vocab)) if max_merges is None else max_merges
        logger.info(f"Entraînement du Tokenizer Sarah Ngin : {num_merges} fusions ciblées...")

        for i in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # Fréquence trop basse pour être utile
                break

            new_idx = len(self.vocab)
            # Concaténation des deux segments d'octets
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_idx] = merged_bytes
            self.inv_vocab[merged_bytes] = new_idx
            self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

            ids_list = self._merge(ids_list, best_pair, new_idx)

        logger.info(f"Tokenizer entraîné avec succès. Taille du vocabulaire finale : {self.vocab_size}")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode une chaîne de caractères en IDs de tokens.
        """
        if not text:
            tokens = []
            if add_bos: tokens = [self.bos_token_id] + tokens
            if add_eos: tokens = tokens + [self.eos_token_id]
            return tokens

        raw_bytes = text.encode("utf-8")
        offset = len(self.special_tokens)
        ids = [offset + b for b in raw_bytes]

        # Application itérative des fusions BPE apprises
        for p0_bytes, p1_bytes in self.merges:
            if len(ids) < 2:
                break
            p0_id = self.inv_vocab[p0_bytes]
            p1_id = self.inv_vocab[p1_bytes]
            merged_id = self.inv_vocab[p0_bytes + p1_bytes]

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

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

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
