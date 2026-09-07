"""
Architecture du Réseau de Neurones Sarah Ngin (Transformer Décodeur Auto-Régressif).
Conception 100% de zéro avec RMSNorm, RoPE (Rotary Position Embeddings) et SwiGLU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import SarahNginConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization pour une exécution ultra-rapide et stable."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Précalcule les fréquences pour l'encodage de position rotatif (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complexe e^(i*freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applique la rotation RoPE sur les tenseurs de Query et Key."""
    # xq: [batch, seq_len, n_heads, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """Mécanisme d'auto-attention causale multi-têtes avec RoPE."""
    def __init__(self, config: SarahNginConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dim = config.dim

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Projections Q, K, V
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Encodage rotatif RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transposition pour calcul d'attention : [batch, n_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Produit scalaire de l'attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)

        output = torch.matmul(scores, xv) # [batch, n_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.dropout(self.wo(output))


class FeedForward(nn.Module):
    """MLP SwiGLU (Gated Linear Unit avec activation SiLU) pour haute expressivité."""
    def __init__(self, config: SarahNginConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.dim) / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Bloc de transformation unifié avec connexions résiduelles et RMSNorm."""
    def __init__(self, layer_id: int, config: SarahNginConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Connexion résiduelle pré-normalisée (Pre-LN style)
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SarahNginTransformer(nn.Module):
    """
    Réseau de Neurones Sarah Ngin complet :
    Modèle de langage génératif autonome optimisé pour appareils mobiles et CPU.
    """
    def __init__(self, config: SarahNginConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        # Plongements lexicaux (Word Embeddings)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Couches Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.n_layers)
        ])

        # Normalisation finale et tête de prédiction
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Partage de poids (Weight Tying) pour économiser la mémoire RAM/VRAM
        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Précalcul des fréquences RoPE
        freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialisation soignée des poids
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Forward pass du modèle.
        input_ids: [batch_size, seq_len]
        targets: [batch_size, seq_len] (optionnel, pour l'entraînement)
        """
        batch_size, seq_len = input_ids.shape

        # Embedding + Dropout
        h = self.dropout(self.tok_embeddings(input_ids))
        
        # Fréquences RoPE pour la séquence active
        freqs_cis = self.freqs_cis[:seq_len]

        # Masque causal triangulaire
        mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        # Propagation à travers les blocs
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # Normalisation finale
        h = self.norm(h)

        # Projection sur le vocabulaire (Logits)
        logits = self.output(h)

        # Calcul de la perte si cibles fournies
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            return logits, loss

        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Retourne le nombre total et actif de paramètres."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_in_megabytes_fp32": (total_params * 4) / (1024 * 1024),
            "size_in_megabytes_int8": (total_params * 1) / (1024 * 1024)
        }
