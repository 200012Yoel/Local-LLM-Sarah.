"""
Protocole d'Entraînement Neuronal Intensif - Sarah Engine (Multi-Langages & Corpus Massif)
Exécution de gradients réels (AdamW, CrossEntropyLoss, Backpropagation complète PyTorch).
"""

import sys
import time
import os
import math
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agent_developer.data_collector import DataCollector
from tokenizer.bpe_tokenizer import SarahTokenizer
from model.config import SarahNginConfig
from model.transformer import SarahNginTransformer
from training.dataset import create_dataloaders
from training.trainer import SarahNginTrainer

def run_intensive_training(epochs=25, batch_size=8, lr=5e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================", flush=True)
    print(f"🚀 PROTOCOLE D'ENTRAÎNEMENT LOURD SARAH ENGINE", flush=True)
    print(f"Hardware assigné : {device.upper()}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Collecte et Synthèse du Corpus
    print("[1/5] Compilation et ingestion intégrale du corpus...", flush=True)
    collector = DataCollector(data_dir="data")
    corpus_path = collector.collect_and_build_all()

    # 2. Tokenizer BPE Multilingue
    print("[2/5] Entraînement du Tokenizer BPE (2048 tokens)...", flush=True)
    tokenizer = SarahTokenizer(vocab_size=2048)
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    tokenizer.train(lines)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    tokenizer.save("checkpoints/sarah_tokenizer.json")

    # 3. Configuration & Modèle Transformer
    print("[3/5] Instanciation du réseau Transformer Sarah Engine (6 Layers / 6 Heads / Dim 192)...", flush=True)
    config = SarahNginConfig.mobile_iphone14(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = 256
    config.dim = 192
    config.n_layers = 6
    config.n_heads = 6
    config.save("checkpoints/sarah_ngin_config.json")

    model = SarahNginTransformer(config).to(device)
    param_info = model.count_parameters()
    print(f"Paramètres totaux : {param_info['total_parameters']:,} | Empreinte FP32 : {param_info['size_in_megabytes_fp32']:.2f} Mo", flush=True)

    # 4. DataLoaders
    print("[4/5] Préparation des tenseurs et flux d'apprentissage...", flush=True)
    train_loader, val_loader = create_dataloaders(
        filepath=str(corpus_path),
        tokenizer=tokenizer,
        block_size=128,
        batch_size=batch_size,
        val_split=0.08
    )

    # 5. Boucle d'Optimisation & Rétropropagation
    print(f"[5/5] DÉMARRAGE DES DESCENTES DE GRADIENT ({epochs} ÉPOQUES)...", flush=True)
    trainer = SarahNginTrainer(model, tokenizer, learning_rate=lr, device=device)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir="checkpoints"
    )

    print("\n✅ ENTRAÎNEMENT LOURD ACHEVÉ : Nouveaux poids certifiés sauvegardés dans checkpoints/sarah_ngin_best.pt", flush=True)

if __name__ == "__main__":
    run_intensive_training(epochs=20, batch_size=8, lr=6e-4)
