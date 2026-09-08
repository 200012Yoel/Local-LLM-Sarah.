"""
Implémentation des couches BitNet b1.58 (Poids Ternaires {-1, 0, 1} et Activations 8-bit).
Straight-Through Estimator (STE) pour la rétropropagation continue.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def round_clip(x, a, b):
    """Arrondi avec écrêtage."""
    return torch.clamp(torch.round(x), a, b)

class BitLinear158(nn.Linear):
    """
    Couche linéaire BitNet b1.58 :
    - Poids quantifiés en ternaire : {-1, 0, 1} via quantification par valeur absolue moyenne
    - Activations quantifiées en 8-bit par normalisation d'échelle
    - Utilise le Straight-Through Estimator (STE) pour conserver le gradient exact en FP32/FP16
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.rms_norm = nn.RMSNorm(in_features, eps=1e-5)

    def weight_quant(self, w: torch.Tensor):
        """
        Quantification ternaire BitNet b1.58 :
        W_quant = RoundClip(W / (gamma + eps), -1, 1) où gamma = mean(abs(W))
        """
        gamma = torch.mean(torch.abs(w)).clamp(min=1e-5)
        w_scaled = w / gamma
        w_quant = round_clip(w_scaled, -1.0, 1.0)
        # STE (Straight-Through Estimator)
        return w + (w_quant - w).detach(), gamma

    def activation_quant(self, x: torch.Tensor):
        """
        Quantification des activations en 8-bit [-128, 127] :
        X_quant = RoundClip(X * (127 / eta), -128, 127) où eta = max(abs(X))
        """
        eta = torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp(min=1e-5)
        scale = 127.0 / eta
        x_scaled = x * scale
        x_quant = round_clip(x_scaled, -128.0, 127.0)
        # STE
        return x + (x_quant - x).detach(), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalisation RMS
        normed_x = self.rms_norm(x)
        
        # 2. Quantification des activations (8-bit)
        x_quant, act_scale = self.activation_quant(normed_x)
        
        # 3. Quantification des poids (ternaire {-1, 0, 1})
        w_quant, weight_gamma = self.weight_quant(self.weight)
        
        # 4. Produit matriciel
        out = F.linear(x_quant, w_quant, bias=None)
        
        # 5. Dé-quantification par produit des échelles d'activation et de poids
        out = (out / act_scale) * weight_gamma
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
