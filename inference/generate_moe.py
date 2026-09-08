"""
Générateur d'Inférence Textuelle Haute Performance pour Sarah Engine MoE BitNet b1.58.
Supporte :
- Inférence Sparse MoE avec Top-K dynamic routing
- Sliding Window Attention (SWA) avec Rolling KV-Cache
- Pénalité de répétition de caractères purs, syllabes et tokens
- Décodage BPE 8k ultra-rapide.
"""

import re
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from pathlib import Path
import logging

from model.transformer_moe import SarahMoETransformer, SarahMoEConfig
from tokenizer.bpe_tokenizer import SarahTokenizer

logger = logging.getLogger(__name__)

class SarahMoEGenerator:
    def __init__(
        self,
        model: SarahMoETransformer,
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

        self._init_token_cache()

    def _init_token_cache(self):
        """Pré-calcule la représentation textuelle des tokens pour le filtrage en temps réel."""
        self.token_str_cache = {}
        for token_id in range(self.tokenizer.vocab_size):
            try:
                self.token_str_cache[token_id] = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                self.token_str_cache[token_id] = ""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[str] = None
    ) -> "SarahMoEGenerator":
        """Instancie un générateur à partir d'un checkpoint MoE BitNet."""
        tokenizer = SarahTokenizer.load(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        config = checkpoint["config"]
        model = SarahMoETransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, tokenizer, device=device)

    def _is_repetition(self, current_text: str, cand_str: str, last_token_str: str = "") -> bool:
        """Pénaliseur strict de répétition de caractères purs, syllabes et tokens."""
        if not cand_str:
            return False

        if last_token_str and cand_str.strip() and cand_str == last_token_str:
            return True

        test_text = (current_text[-40:] + cand_str) if len(current_text) > 40 else (current_text + cand_str)

        if re.search(r'(.{2,15})\1$', test_text):
            return True

        if re.search(r'([A-Z0-9:;,=_\-!#])\1', test_text):
            return True

        if re.search(r'([a-zà-ÿ])\1{2,}', test_text):
            return True

        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.45,
        top_k: int = 30,
        top_p: float = 0.85,
        repetition_penalty: float = 1.35,
        presence_penalty: float = 0.3,
        no_repeat_ngram_size: int = 3,
        stop_sequences: Optional[List[str]] = None,
        min_new_tokens: int = 4
    ) -> str:
        """Génère du texte de manière fluide et contrôlée avec le modèle MoE BitNet."""
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated_tokens = []
        recent_window = 40
        current_text = prompt
        last_token_str = ""

        default_stops = ["<eos>", "<pad>", "\n\n", "Question:", "User:", "Utilisateur:"]
        if stop_sequences:
            default_stops.extend(stop_sequences)

        with torch.no_grad():
            for step_idx in range(max_new_tokens):
                if input_tensor.shape[1] > self.model.config.max_seq_len:
                    input_tensor = input_tensor[:, -self.model.config.max_seq_len:]

                logits, _, _ = self.model(input_tensor)
                next_token_logits = logits[0, -1, :].clone()

                # 1. Pénalité de répétition de tokens
                if generated_tokens and repetition_penalty > 1.0:
                    token_counts = {}
                    window_tokens = generated_tokens[-recent_window:]
                    for t in window_tokens:
                        token_counts[t] = token_counts.get(t, 0) + 1

                    for token_id, count in token_counts.items():
                        penalty = (repetition_penalty ** count) + (presence_penalty * count)
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= penalty
                        else:
                            next_token_logits[token_id] /= penalty

                # 2. Blocage des n-grammes répétés
                if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size:
                    ngram = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])
                    for i in range(len(generated_tokens) - no_repeat_ngram_size + 1):
                        prev_ngram = tuple(generated_tokens[i : i + no_repeat_ngram_size - 1])
                        if prev_ngram == ngram:
                            forbidden_token = generated_tokens[i + no_repeat_ngram_size - 1]
                            next_token_logits[forbidden_token] = -float("Inf")

                # 3. Pénaliseur de répétition de caractères purs
                top_cands = torch.topk(next_token_logits, min(60, next_token_logits.size(-1))).indices.tolist()
                for cand_id in top_cands:
                    cand_str = self.token_str_cache.get(cand_id, "")
                    if self._is_repetition(current_text, cand_str, last_token_str):
                        next_token_logits[cand_id] = -float("Inf")

                # 4. Échantillonnage / Lissage
                if temperature > 0:
                    scaled_logits = next_token_logits / max(temperature, 1e-4)

                    if top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[-1]] = -float("Inf")

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        scaled_logits[indices_to_remove] = -float("Inf")

                    probs = F.softmax(scaled_logits, dim=-1)
                    
                    if torch.isnan(probs).any() or (probs == 0).all():
                        break

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                token_val = next_token.item()

                # 5. Arrêt strict sur EOS
                if token_val in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                    break

                generated_tokens.append(token_val)
                token_text = self.token_str_cache.get(token_val, "")
                last_token_str = token_text
                current_text += token_text
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

                # 6. Stop sequences
                stop_triggered = False
                for stop_seq in default_stops:
                    if stop_seq in current_text[len(prompt):]:
                        stop_triggered = True
                        break
                if stop_triggered:
                    break

                # 7. Fin de phrase
                gen_so_far = current_text[len(prompt):].strip()
                if step_idx >= min_new_tokens and gen_so_far.endswith((".", "!", "?")):
                    if len(gen_so_far.split()) >= 6:
                        break

        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for stop_seq in default_stops:
            if stop_seq in final_text:
                final_text = final_text.split(stop_seq)[0]

        return re.sub(r'\s+', ' ', final_text).strip()
