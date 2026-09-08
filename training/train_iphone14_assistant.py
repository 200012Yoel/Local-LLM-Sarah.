"""
Pipeline d'Entraînement Maître Conversationnel (Mini-ChatGPT From Scratch)
pour Sarah Ngin - Architecture Optimale pour iPhone 14 (~18.6M Paramètres).
Optimisé pour une exécution multi-cœurs rapide avec Target Masking.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Optimiser l'utilisation CPU multi-cœurs
torch.set_num_threads(min(8, os.cpu_count() or 4))

from model.config import SarahNginConfig
from model.transformer import SarahNginTransformer
from tokenizer.bpe_tokenizer import SarahTokenizer
from training.dataset import ChatInstructDataset, create_chat_dataloaders
from inference.generate import SarahNginGenerator

def train_iphone14_assistant(
    dataset_path: str = "data/conversational/chat_instruct_dataset.json",
    tokenizer_path: str = "checkpoints/sarah_tokenizer_8k.json",
    checkpoint_out: str = "checkpoints/sarah_iphone14_best.pt",
    epochs: int = 15,
    batch_size: int = 16,
    max_seq_len: int = 128,
    learning_rate: float = 6e-4,
    device_str: str = "cpu"
):
    print("=" * 75, flush=True)
    print("  ENTRAÎNEMENT DU MODÈLE CONVERSATIONNEL SARAH NGIN POUR IPHONE 14", flush=True)
    print("=" * 75, flush=True)

    device = torch.device(device_str)
    
    # 1. Charger le Tokenizer
    tokenizer = SarahTokenizer.load(tokenizer_path)
    print(f"[Tokenizer] Vocabulaire actif : {tokenizer.vocab_size} tokens", flush=True)
    print(f"[Rôles Spéciaux] <bos>={tokenizer.bos_token_id}, <eos>={tokenizer.eos_token_id}, <user>={tokenizer.user_token_id}, <assistant>={tokenizer.assistant_token_id}", flush=True)

    # 2. Préparer les DataLoaders avec Target Masking
    print("[Données] Construction des séquences conversationnelles avec Target Masking...", flush=True)
    train_loader, val_loader = create_chat_dataloaders(
        dataset_filepath=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        val_split=0.1
    )
    print(f"[Données] Échantillons Train : {len(train_loader.dataset):,} | Échantillons Val : {len(val_loader.dataset):,}", flush=True)

    # 3. Architecture Transformer Optimale iPhone 14
    config = SarahNginConfig(
        vocab_size=tokenizer.vocab_size,
        dim=384,
        n_layers=8,
        n_heads=12,
        max_seq_len=max_seq_len,
        dropout=0.05,
        tie_word_embeddings=True
    )
    config.save("checkpoints/sarah_iphone14_config.json")

    model = SarahNginTransformer(config).to(device)
    params_info = model.count_parameters()
    print(f"[Architecture] Paramètres totaux : {params_info['total_parameters']:,} (~{params_info['size_in_megabytes_fp32']:.2f} Mo FP32 / ~{params_info['size_in_megabytes_int8']:.2f} Mo INT8)", flush=True)

    # 4. Optimiseur & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=learning_rate * 0.1)

    best_val_loss = float("inf")
    loss_history = []
    start_time = time.time()

    # 5. Boucle d'Entraînement
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / max(1, train_steps)

        # Validation Loss
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                _, vloss = model(vx, targets=vy)
                total_val_loss += vloss.item()
                val_steps += 1
        avg_val_loss = total_val_loss / max(1, val_steps) if val_steps > 0 else avg_train_loss

        elapsed = time.time() - start_time
        print(f"Époque [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Temps: {elapsed:.1f}s", flush=True)

        loss_history.append(round(avg_train_loss, 4))

        # Sauvegarde du Meilleur Checkpoint
        if avg_val_loss < best_val_loss or epoch == epochs:
            best_val_loss = avg_val_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "vocab_size": tokenizer.vocab_size,
                "target_hardware": "Apple iPhone 14 (A15 Bionic)"
            }
            torch.save(ckpt_dict, checkpoint_out)
            torch.save(ckpt_dict, "checkpoints/sarah_ngin_master.pt")
            torch.save(ckpt_dict, "checkpoints/sarah_ngin_best.pt")

        # Mise à jour du fichier de progression pour le Web UI
        with open("training/training_progress.json", "w", encoding="utf-8") as pf:
            json.dump({
                "status": "completed" if epoch == epochs else "training",
                "current_epoch": epoch,
                "total_epochs": epochs,
                "percent_complete": int((epoch / epochs) * 100),
                "current_loss": round(avg_train_loss, 4),
                "best_loss": round(best_val_loss, 4),
                "loss_history": loss_history,
                "parameters": params_info["total_parameters"],
                "target_hardware": "Apple iPhone 14 (A15 Bionic)",
                "last_log": f"Époque {epoch}/{epochs} validée (Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f})",
                "elapsed_seconds": round(elapsed, 1)
            }, pf, indent=2)

    print("=" * 75, flush=True)
    print(f" Entraînement terminé ! Checkpoint maître sauvegardé : {checkpoint_out}", flush=True)
    print(f" Meilleure Validation Loss : {best_val_loss:.4f}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    train_iphone14_assistant(
        epochs=15,
        batch_size=16,
        max_seq_len=128,
        learning_rate=6e-4
    )
