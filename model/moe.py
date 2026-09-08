"""
Architecture Mixture of Experts (MoE) pour Sarah Engine.
Routage dynamique Top-K avec Router Gating intelligent et pénalité d'équilibrage de charge.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from model.bitnet import BitLinear158

class BitNetExpert(nn.Module):
    """Un expert Feed-Forward utilisant des couches BitLinear ternaires b1.58."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = BitLinear158(dim, hidden_dim, bias=False)
        self.w2 = BitLinear158(hidden_dim, dim, bias=False)
        self.w3 = BitLinear158(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Architecture SwiGLU avec poids ternaires
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TopKRouter(nn.Module):
    """
    Routeur Top-K intelligent avec calcul de la perte d'équilibrage de charge (Load Balancing Loss)
    et élimination des chemins neuronaux aberrants.
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        # Bruit gaussien pour l'exploration équilibrée des experts en phase d'apprentissage
        self.noise_linear = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor, train_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, dim]
        logits = self.gate(x)
        
        if self.training and train_noise:
            # Ajout de bruit contrôlé pour stabiliser l'allocation des experts
            noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
            logits = logits + noise

        # Softmax sur tous les experts
        gates = F.softmax(logits, dim=-1)
        
        # Sélection Top-K
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Renormalisation des poids Top-K
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Calcul de la perte auxiliaire d'équilibrage de charge (Switch Transformer / MoE aux loss)
        # P_i: fraction de probabilité allouée à l'expert i
        # f_i: fraction de tokens routés vers l'expert i
        # Aux_Loss = num_experts * sum(f_i * P_i)
        flat_gates = gates.view(-1, self.num_experts)
        P = flat_gates.mean(dim=0)
        
        flat_indices = top_k_indices.view(-1)
        # Comptage de la fréquence de chaque expert
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (flat_indices == i).float().sum()
        f = f / (flat_indices.numel() + 1e-6)
        
        aux_loss = self.num_experts * torch.sum(P * f)
        
        return top_k_gates, top_k_indices, aux_loss

class MoELayer(nn.Module):
    """
    Couche Sparse Mixture of Experts complète :
    - N experts ternaires BitNet b1.58
    - Routeur Top-K dynamique
    - Calcul de l'équilibrage et routage optimisé
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(dim, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            BitNetExpert(dim, hidden_dim, dropout=dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        # Flatten pour le routage
        flat_x = x.reshape(-1, dim)
        
        # Obtenir les pondérations et indices des experts
        top_k_gates, top_k_indices, aux_loss = self.router(flat_x)
        
        # Accumulateur pour la sortie
        final_output = torch.zeros_like(flat_x)
        
        # Traitement par expert
        for i, expert in enumerate(self.experts):
            # Masque des tokens assignés à cet expert
            expert_mask = (top_k_indices == i)
            # Y a-t-il au moins un token assigné à cet expert ?
            if expert_mask.any():
                token_indices, k_positions = torch.where(expert_mask)
                selected_x = flat_x[token_indices]
                expert_out = expert(selected_x)
                
                # Pondérer la sortie de l'expert par le score du routeur
                gates_for_expert = top_k_gates[token_indices, k_positions].unsqueeze(-1)
                final_output.index_add_(0, token_indices, expert_out * gates_for_expert)

        output = final_output.view(batch_size, seq_len, dim)
        return output, aux_loss
