"""
Boucle d'entraînement avancée pour le modèle d'IA Sarah Ngin.
Gère l'optimisation, la descente de gradient, l'ordonnancement du taux d'apprentissage et le monitoring.
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import logging

from model.transformer import SarahNginTransformer
from model.config import SarahNginConfig
from tokenizer.bpe_tokenizer import SarahTokenizer

logging.basicConfig(level=logging.INFO, format="[Sarah Ngin Trainer] %(message)s")
logger = logging.getLogger(__name__)

class SarahNginTrainer:
    def __init__(
        self,
        model: SarahNginTransformer,
        tokenizer: SarahTokenizer,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Sélection automatique du hardware
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
        logger.info(f"Modèle assigné sur le matériel : {self.device}")

        # Configuration de l'optimiseur AdamW avec découplage du Weight Decay
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        self.learning_rate = learning_rate

    def get_lr(self, step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> float:
        """Calcul du taux d'apprentissage avec montée linéaire (warmup) et décroissance cosinus."""
        if step < warmup_steps:
            return self.learning_rate * (step + 1) / max(1, warmup_steps)
        if step > max_steps:
            return self.learning_rate * min_lr_ratio
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.learning_rate * (min_lr_ratio + coeff * (1.0 - min_lr_ratio))

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Évalue la fonction de perte et la perplexité sur le jeu de validation."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        perplexity = math.exp(min(avg_loss, 20)) # Évite l'overflow sur perplexité
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        warmup_steps: int = 20,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints"
    ):
        """Boucle d'entraînement principale."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        total_steps = len(train_loader) * epochs
        step = 0
        best_loss = float("inf")
        start_time = time.time()

        logger.info(f"=== Début de l'entraînement Sarah Ngin ===")
        logger.info(f"Époques : {epochs} | Steps totaux : {total_steps} | Batch size : {train_loader.batch_size}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                # Mise à jour dynamique du Learning Rate (Warmup + Cosine)
                lr = self.get_lr(step, warmup_steps, total_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits, loss = self.model(x, targets=y)
                loss.backward()

                # Gradient Clipping pour stabiliser l'apprentissage
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                running_loss += loss.item()
                step += 1

                if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                    current_loss = running_loss / (batch_idx + 1)
                    ppl = math.exp(min(current_loss, 20))
                    logger.info(f"Époque [{epoch}/{epochs}] Step [{batch_idx+1}/{len(train_loader)}] - Loss: {current_loss:.4f} | Perplexité: {ppl:.2f} | LR: {lr:.6f}")

            avg_train_loss = running_loss / len(train_loader)
            metrics_msg = f"Époque {epoch} terminée - Train Loss: {avg_train_loss:.4f}"

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics_msg += f" | Val Loss: {val_metrics['val_loss']:.4f} | Val PPL: {val_metrics['val_perplexity']:.2f}"
                current_eval_loss = val_metrics["val_loss"]
            else:
                current_eval_loss = avg_train_loss

            logger.info(metrics_msg)

            # Sauvegarde du meilleur modèle
            if current_eval_loss < best_loss:
                best_loss = current_eval_loss
                self.save_checkpoint(str(checkpoint_path / "sarah_ngin_best.pt"))

        total_duration = time.time() - start_time
        logger.info(f"Entraînement terminé avec succès en {total_duration:.2f} secondes !")
        # Sauvegarde finale
        self.save_checkpoint(str(checkpoint_path / "sarah_ngin_final.pt"))

    def save_checkpoint(self, filepath: str):
        """Sauvegarde les poids du modèle et la configuration associée."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config,
            "vocab_size": self.tokenizer.vocab_size
        }
        torch.save(state, filepath)
        logger.info(f"Checkpoint sauvegardé : {filepath}")
