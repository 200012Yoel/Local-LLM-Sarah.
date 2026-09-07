"""
Module de chargement et de découpage des données pour l'entraînement de Sarah Ngin.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path

from tokenizer.bpe_tokenizer import SarahTokenizer

class TextDataset(Dataset):
    """
    Dataset PyTorch avec fenêtrage glissant pour l'apprentissage de la logique et de la langue.
    """
    def __init__(self, token_ids: List[int], block_size: int = 128, stride: Optional[int] = None):
        self.block_size = block_size
        self.stride = stride or (block_size // 2)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        
        # Liste des indices de départ avec le stride
        self.indices = list(range(0, max(0, len(self.data) - self.block_size), self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def build_token_stream(filepath: str, tokenizer: SarahTokenizer) -> List[int]:
    """Lit un fichier texte et le convertit en flux continu de tokens avec séparateurs <eos>."""
    path = Path(filepath)
    tokens: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            encoded = tokenizer.encode(line_str, add_bos=True, add_eos=True)
            tokens.extend(encoded)
    return tokens


def create_dataloaders(
    filepath: str,
    tokenizer: SarahTokenizer,
    block_size: int = 128,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Crée les DataLoaders d'entraînement et de validation."""
    all_tokens = build_token_stream(filepath, tokenizer)
    split_idx = int(len(all_tokens) * (1.0 - val_split))
    
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:] if val_split > 0 else []

    train_dataset = TextDataset(train_tokens, block_size=block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = None
    if len(val_tokens) > block_size:
        val_dataset = TextDataset(val_tokens, block_size=block_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

    return train_loader, val_loader
