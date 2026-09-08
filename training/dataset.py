"""
Module de chargement et de préparation des données pour Sarah Ngin.
Implémente le formatage conversationnel (<bos>, <user>, <assistant>, <eos>)
avec Masquage d'Instruction (Target Masking / ignore_index=-100)
pour un apprentissage de type Instruct / Mini-ChatGPT.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from tokenizer.bpe_tokenizer import SarahTokenizer

class ChatInstructDataset(Dataset):
    """
    Dataset conversationnel avec masquage de cibles (Target Masking).
    La perte (Cross-Entropy) n'est calculée QUE sur les tokens de l'assistant.
    Les tokens de l'utilisateur sont masqués avec la valeur -100.
    """
    def __init__(
        self,
        dialogues: List[List[Dict[str, str]]],
        tokenizer: SarahTokenizer,
        max_seq_len: int = 512,
        pad_to_max_len: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self._process_dialogues(dialogues, pad_to_max_len)

    def _process_dialogues(self, dialogues: List[List[Dict[str, str]]], pad_to_max_len: bool):
        for dialogue in dialogues:
            full_input_ids: List[int] = [self.tokenizer.bos_token_id]
            full_labels: List[int] = [-100]  # On ne prédit pas <bos>

            for turn in dialogue:
                role = turn.get("role", "user")
                content = turn.get("content", "").strip()
                if not content:
                    continue

                if role == "user":
                    # Encoder le tour utilisateur : <user> content
                    turn_tokens = [self.tokenizer.user_token_id] + self.tokenizer.encode(content, add_bos=False, add_eos=False)
                    full_input_ids.extend(turn_tokens)
                    # Masquer les cibles pour l'utilisateur
                    full_labels.extend([-100] * len(turn_tokens))

                elif role == "assistant":
                    # Encoder le tour assistant : <assistant> content <eos>
                    turn_tokens = [self.tokenizer.assistant_token_id] + self.tokenizer.encode(content, add_bos=False, add_eos=False) + [self.tokenizer.eos_token_id]
                    full_input_ids.extend(turn_tokens)
                    # Prédire les tokens de l'assistant !
                    full_labels.extend(turn_tokens)

            if len(full_input_ids) < 4:
                continue

            # Tronquer à max_seq_len + 1 pour créer (x, y)
            if len(full_input_ids) > self.max_seq_len + 1:
                full_input_ids = full_input_ids[: self.max_seq_len + 1]
                full_labels = full_labels[: self.max_seq_len + 1]

            # x = tokens[:-1], y = labels[1:]
            x_ids = full_input_ids[:-1]
            y_ids = full_labels[1:]

            if pad_to_max_len and len(x_ids) < self.max_seq_len:
                pad_len = self.max_seq_len - len(x_ids)
                x_ids = x_ids + [self.tokenizer.pad_token_id] * pad_len
                y_ids = y_ids + [-100] * pad_len

            self.samples.append((
                torch.tensor(x_ids, dtype=torch.long),
                torch.tensor(y_ids, dtype=torch.long)
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def create_chat_dataloaders(
    dataset_filepath: str,
    tokenizer: SarahTokenizer,
    max_seq_len: int = 512,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Charge le dataset JSON conversationnel et crée les DataLoaders d'entraînement et validation."""
    with open(dataset_filepath, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    split_idx = int(len(dialogues) * (1.0 - val_split))
    train_dialogues = dialogues[:split_idx]
    val_dialogues = dialogues[split_idx:]

    train_dataset = ChatInstructDataset(train_dialogues, tokenizer, max_seq_len=max_seq_len)
    val_dataset = ChatInstructDataset(val_dialogues, tokenizer, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader

# Compatibilité ascendante
class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int = 128, stride: Optional[int] = None):
        self.block_size = block_size
        self.stride = stride or (block_size // 2)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.indices = list(range(0, max(0, len(self.data) - self.block_size), self.stride))

    def __len__(self) -> int:
        return max(1, len(self.indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.indices:
            pad = torch.zeros(self.block_size, dtype=torch.long)
            return pad, pad
        start = self.indices[idx]
        chunk = self.data[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:-1], chunk[1:]

def create_dataloaders(filepath: str, tokenizer: SarahTokenizer, block_size: int = 128, batch_size: int = 16, val_split: float = 0.1, max_lines: Optional[int] = 3000, num_workers: int = 0):
    return create_chat_dataloaders("data/conversational/chat_instruct_dataset.json", tokenizer, max_seq_len=block_size, batch_size=batch_size, val_split=val_split, num_workers=num_workers)

