"""
Architecture Transformer BitNet MoE (Mixture of Experts) pour Sarah Engine.
Combine :
- Poids Ternaires BitNet b1.58
- Sparse MoE avec Top-K Routing
- Embeddings Factorisés (V -> d_embed -> d_model)
- Sliding Window Attention (SWA) pour plafonner l'usage mémoire RAM (< 300 Mo / iPhone 14).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict

from model.bitnet import BitLinear158
from model.moe import MoELayer

@dataclass
class SarahMoEConfig:
    vocab_size: int = 8192          # Vocabulaire étendu 8k tokens multilingue & code
    dim: int = 256                  # Dimension cachée principale
    embed_dim: int = 64             # Dimension factorisée d'embedding
    n_layers: int = 6               # Nombre de blocs Transformer MoE
    n_heads: int = 8                # Nombre de têtes d'attention
    num_experts: int = 8            # Nombre d'experts par couche
    top_k_experts: int = 2          # Nombre d'experts actifs par token
    hidden_dim_mult: float = 1.5    # Multiplicateur dimension cachée FFN
    sliding_window: int = 256       # Fenêtre glissante pour l'attention (SWA)
    max_seq_len: int = 512          # Longueur maximale
    norm_eps: float = 1e-5
    dropout: float = 0.05
    aux_loss_coef: float = 0.02     # Coefficient de la perte auxiliaire d'équilibrage

class FactorizedEmbedding(nn.Module):
    """
    Embedding factorisé : V (8192) x embed_dim (64) -> projection vers dim (256).
    Compresse la table d'embedding de plus de 4x.
    """
    def __init__(self, vocab_size: int, embed_dim: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x))

class RotaryEmbedding(nn.Module):
    """Embeddings positionnels rotatifs (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D_h]
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return (x * cos) + (rotated * sin)

class SlidingWindowAttention(nn.Module):
    """
    Attention multi-têtes avec Sliding Window Attention (SWA)
    et projections quantifiées BitLinear158.
    """
    def __init__(self, config: SarahMoEConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.sliding_window = config.sliding_window

        self.q_proj = BitLinear158(config.dim, config.dim, bias=False)
        self.k_proj = BitLinear158(config.dim, config.dim, bias=False)
        self.v_proj = BitLinear158(config.dim, config.dim, bias=False)
        self.out_proj = BitLinear158(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Application de RoPE
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Gestion du Rolling KV-Cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            # Tronquer le cache à la taille de la fenêtre glissante pour borner la RAM
            if k.shape[2] > self.sliding_window:
                k = k[:, :, -self.sliding_window:, :]
                v = v[:, :, -self.sliding_window:, :]

        new_kv_cache = (k, v)
        kv_len = k.shape[2]

        # Masque causal avec fenêtre glissante
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Masque SWA
        if S > 1:
            q_indices = torch.arange(S, device=x.device).unsqueeze(1) + (kv_len - S)
            k_indices = torch.arange(kv_len, device=x.device).unsqueeze(0)
            distance = q_indices - k_indices
            
            # Causal mask (distance < 0) ET sliding window (distance >= sliding_window)
            invalid_mask = (distance < 0) | (distance >= self.sliding_window)
            scores = scores.masked_fill(invalid_mask.unsqueeze(0).unsqueeze(0), -float("inf"))

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, self.dim)
        out = self.out_proj(context)
        return out, new_kv_cache

class SarahMoEBlock(nn.Module):
    """Un bloc Transformer complet associant SWA et Sparse MoE."""
    def __init__(self, config: SarahMoEConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = SlidingWindowAttention(config)
        self.norm2 = nn.RMSNorm(config.dim, eps=config.norm_eps)
        
        hidden_dim = int(config.dim * config.hidden_dim_mult)
        self.moe = MoELayer(
            dim=config.dim,
            hidden_dim=hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Attention résiduelle
        attn_out, new_cache = self.attn(self.norm1(x), rope_cos, rope_sin, kv_cache=kv_cache)
        x = x + attn_out
        
        # 2. Sparse MoE résiduelle
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, aux_loss, new_cache

class SarahMoETransformer(nn.Module):
    """
    Réseau de Neurones Sarah Engine MoE BitNet b1.58.
    Capacité nominale étendue et ultra-compact mobile (< 300 Mo sur iPhone 14).
    """
    def __init__(self, config: SarahMoEConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = FactorizedEmbedding(config.vocab_size, config.embed_dim, config.dim)
        self.rope = RotaryEmbedding(config.dim // config.n_heads, max_seq_len=config.max_seq_len)
        
        self.layers = nn.ModuleList([
            SarahMoEBlock(config) for _ in range(config.n_layers)
        ])
        self.norm_f = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S = tokens.shape
        x = self.tok_embeddings(tokens)
        rope_cos, rope_sin = self.rope(x, S)
        
        total_aux_loss = torch.tensor(0.0, device=tokens.device)
        new_caches = []

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, aux_loss, new_cache = layer(x, rope_cos, rope_sin, kv_cache=layer_cache)
            total_aux_loss = total_aux_loss + aux_loss
            new_caches.append(new_cache)

        x = self.norm_f(x)
        logits = self.head(x)
        return logits, total_aux_loss, new_caches

    def count_parameters(self) -> dict:
        """Calcule les paramètres totaux et les paramètres actifs par token."""
        total_params = sum(p.numel() for p in self.parameters())
        # Paramètres actifs par token (seuls top_k experts sont activés)
        active_params = total_params - sum(
            p.numel() for layer in self.layers for expert in layer.moe.experts[self.config.top_k_experts:] for p in expert.parameters()
        )
        return {
            "total_parameters": total_params,
            "active_parameters_per_token": active_params,
            "experts_count": self.config.num_experts,
            "active_experts": self.config.top_k_experts,
            "vocab_size": self.config.vocab_size
        }
