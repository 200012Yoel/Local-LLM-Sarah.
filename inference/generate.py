"""
Moteur de Génération et d'Inférence Textuelle pour Sarah Ngin.
Supporte Temperature, Top-K, Top-P (Nucleus Sampling) et Pénalité de Répétition.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Generator
from pathlib import Path
import logging

from model.transformer import SarahNginTransformer
from model.config import SarahNginConfig
from tokenizer.bpe_tokenizer import SarahTokenizer

logger = logging.getLogger(__name__)

class SarahNginGenerator:
    def __init__(
        self,
        model: SarahNginTransformer,
        tokenizer: SarahTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[str] = None
    ) -> "SarahNginGenerator":
        """Instancie un générateur à partir d'un checkpoint et d'un tokenizer."""
        tokenizer = SarahTokenizer.load(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        config = checkpoint["config"]
        model = SarahNginTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, tokenizer, device=device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stream: bool = False
    ) -> str:
        """
        Génère une suite logique à partir d'un prompt texte.
        """
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tronquer si la taille de la séquence dépasse max_seq_len
                if input_tensor.shape[1] > self.model.max_seq_len:
                    input_tensor = input_tensor[:, -self.model.max_seq_len:]

                out = self.model(input_tensor)
                logits = out[0] if isinstance(out, tuple) else out
                next_token_logits = logits[0, -1, :].clone()

                # Application de la pénalité de répétition
                if repetition_penalty != 1.0:
                    for token_id in set(input_tensor[0].tolist()):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty

                # Température
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                    # Filtrage Top-K
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[-1]] = -float("Inf")

                    # Filtrage Top-P (Nucleus Sampling)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float("Inf")

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy search si temperature == 0
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                token_val = next_token.item()

                # Arrêt si le token EOS est atteint
                if token_val == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(token_val)
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        # Décodage final
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
