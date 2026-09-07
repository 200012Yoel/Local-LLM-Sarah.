"""
Configuration des Hyperparamètres du Modèle Sarah Ngin.
Optimisé pour une empreinte mémoire réduite (< 4 Go RAM / iPhone 14).
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class SarahNginConfig:
    # Architecture
    vocab_size: int = 4096          # Taille du vocabulaire BPE multilingue
    dim: int = 256                  # Dimension cachée (Embedding / Hidden dimension)
    n_layers: int = 6               # Nombre de couches Transformer
    n_heads: int = 8                # Nombre de têtes d'attention
    n_kv_heads: int = 8             # Multi-Query / Grouped-Query attention
    multiple_of: int = 64           # Alignement de dimension pour SwiGLU FFN
    ffn_dim_multiplier: float = 1.3 # Multiplicateur de dimension MLP
    norm_eps: float = 1e-5          # Epsilon pour RMSNorm
    max_seq_len: int = 512          # Longueur maximale de séquence contextuelle
    dropout: float = 0.1            # Taux de régularisation dropout
    tie_word_embeddings: bool = True # Partage des poids Embedding <-> Output Head

    def save(self, filepath: str):
        """Sauvegarde la configuration en format JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SarahNginConfig":
        """Charge la configuration depuis un fichier JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def mobile_iphone14(cls, vocab_size: int = 4096) -> "SarahNginConfig":
        """Profil Ultra-Léger calibré pour iPhone 14 et appareils mobiles (< 500 Mo RAM)."""
        return cls(
            vocab_size=vocab_size,
            dim=256,
            n_layers=6,
            n_heads=8,
            max_seq_len=512,
            dropout=0.05,
            tie_word_embeddings=True
        )

    @classmethod
    def standard_edge(cls, vocab_size: int = 4096) -> "SarahNginConfig":
        """Profil Standard haute précision pour CPU/GPU et Edge devices (< 1.5 Go RAM)."""
        return cls(
            vocab_size=vocab_size,
            dim=384,
            n_layers=8,
            n_heads=12,
            max_seq_len=1024,
            dropout=0.1,
            tie_word_embeddings=True
        )
