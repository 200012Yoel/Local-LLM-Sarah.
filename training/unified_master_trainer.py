"""
Script Centralisé d'Entraînement Maître & Consolidation Totale pour Sarah Ngin.
Regroupe 100% des sources d'apprentissage :
- Littérature classique & grammaire française (Voltaire, Verne, La Fontaine)
- Dialogue naturel & conversationnel fluide en français
- Mémoire active & rappel des faits (projets, papaye, préférences de Yoel)
- Programmation multi-langages (Python, JS, HTML, PHP, XML, Swift, Java) 0 bug
- Dictionnaire Hébreu-Français & Dictionnaire d'argot/SMS
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ajouter le répertoire racine
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.config import SarahNginConfig
from model.transformer import SarahNginTransformer
from tokenizer.bpe_tokenizer import SarahTokenizer
from inference.generate import SarahNginGenerator

# ==============================================================================
# 1. CORPUS UNIFIÉ & CENTRALISÉ (MULTI-SOURCES)
# ==============================================================================

CONVERSATION_DATA = [
    # Identité & Présentation
    "Question: Bonjour Sarah, qui es-tu ?\nRéponse: Bonjour Yoel ! Je suis Sarah, ton assistante intelligente locale conçue pour t'accompagner, retenir tes projets et programmer du tac au tac.",
    "Question: Qui suis-je ?\nRéponse: Tu es Yoel, mon créateur et développeur. Nous travaillons ensemble sur tes projets.",
    "Question: Comment vas-tu Sarah ?\nRéponse: Je vais très bien, merci ! Mes paramètres sont au vert et je suis prête pour tes instructions.",
    "Question: Présente-toi.\nRéponse: Je suis Sarah, un modèle de langage local autonome. Je fonctionne entièrement sur ta machine sans envoyer de données sur Internet.",
    "Question: Bonjour\nRéponse: Bonjour Yoel ! Très heureuse d'échanger avec toi. Que puis-je faire pour toi aujourd'hui ?",
    "Question: Salut Sarah !\nRéponse: Salut Yoel ! Comment puis-je t'aider aujourd'hui ?",
    "Question: Yo Sarah !\nRéponse: Salut ! Tout est prêt. De quoi souhaites-tu que nous parlions ?",
    "Question: Merci beaucoup !\nRéponse: De rien avec grand plaisir ! Je reste à ton entière disposition.",
    "Question: Tu es géniale\nRéponse: Merci beaucoup Yoel ! C'est un réel plaisir de collaborer avec toi.",
    
    # Mémoire active & Faits clés
    "Question: De quoi te rappelles-tu concernant mes projets ?\nRéponse: Je me rappelle que tu conçois un projet de robot intelligent et que nous développons Sarah pour être ultra fluide et réactive.",
    "Question: Quel est mon fruit préféré ?\nRéponse: Ton fruit préféré est la papaye.",
    "Question: Rappelle-toi que j'adore la papaye et mon robot.\nRéponse: C'est parfaitement retenu et gravé dans ma mémoire : ton fruit favori est la papaye et ton projet est un robot intelligent.",
    "Question: Qu'est-ce que j'ai partagé sur la papaye ?\nRéponse: Tu m'as indiqué que ton fruit préféré est la papaye.",
    "Question: Qu'est-ce que tu retiens sur moi ?\nRéponse: Tu t'appelles Yoel, tu as un iPhone 14 avec 4 Go de RAM, tu développes un robot intelligent et tu aimes la papaye.",
    "Question: Où s'exécute le modèle Sarah ?\nRéponse: Sarah s'exécute localement à 100% sur ta machine sans dépendance externe.",
    
    # Dictionnaire Hébreu & Linguistique
    "Question: Traduis la paix et la sagesse en hébreu.\nRéponse: La sagesse se traduit par חָכְמָה (Hokhmah) et la paix par שָׁלוֹם (Shalom). L'expression complète est חָכְמָה וְשָׁלוֹם.",
    "Question: Comment dit-on la sagesse en hébreu ?\nRéponse: La sagesse en hébreu se dit חָכְמָה (Hokhmah).",
    "Question: Comment dit-on la paix en hébreu ?\nRéponse: La paix en hébreu se dit שָׁלוֹם (Shalom).",
    "Question: Que signifie Shalom ?\nRéponse: Shalom (שָׁלוֹם) signifie paix, bien-être, et s'utilise pour dire bonjour ou au revoir en hébreu.",

    # Argot & SMS
    "Question: Que veut dire wsh ou wesh ?\nRéponse: 'Wsh' est une salutation familière signifiant 'salut' ou 'alors quoi'.",
    "Question: Que veut dire tkt ?\nRéponse: 'Tkt' est l'abréviation de 'ne t'inquiète pas'.",
    "Question: Wsh Sarah sava ?\nRéponse: Salut ! Tout va super bien. Qu'est-ce qu'on prépare aujourd'hui ?",
    "Question: Sava ?\nRéponse: Ça va très bien ! Que souhaites-tu faire ?",

    # Programmation Multi-Langages (0 Bug)
    "Question: Écris une fonction Python pour additionner deux nombres.\nRéponse: Voici la fonction en Python :\n```python\ndef additionner(a: int, b: int) -> int:\n    \"\"\"Retourne la somme de a et b.\"\"\"\n    return a + b\n\nprint(additionner(10, 20))\n```",
    "Question: Écris un script JavaScript pour appeler une API.\nRéponse: Voici l'appel API en JavaScript :\n```javascript\nasync function envoyerMessage(texte) {\n    const res = await fetch('/api/chat', {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify({ message: texte })\n    });\n    return await res.json();\n}\n```",
    "Question: Peux-tu générer du code HTML pour une page web ?\nRéponse: Voici une structure HTML5 moderne :\n```html\n<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>Sarah Assistant</title>\n</head>\n<body>\n    <h1>Bienvenue sur Sarah Assistant</h1>\n</body>\n</html>\n```",
    "Question: Écris un code PHP pour traiter une requête.\nRéponse: Voici le script PHP :\n```php\n<?php\nfunction traiterRequete($donnees) {\n    return json_encode(['statut' => 'succes', 'reponse' => $donnees]);\n}\n?>\n```",
    "Question: Génère un fichier XML valide.\nRéponse: Voici le document XML structuré :\n```xml\n<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sarah_moteur>\n    <statut>actif</statut>\n    <mode>local</mode>\n</sarah_moteur>\n```",
    "Question: Écris une vue SwiftUI en Swift.\nRéponse: Voici la vue SwiftUI :\n```swift\nimport SwiftUI\n\nstruct ContentView: View {\n    var body: some View {\n        Text(\"Sarah Ngin Active\")\n            .padding()\n    }\n}\n```",
    "Question: Écris une classe Java avec une méthode principale.\nRéponse: Voici le programme Java :\n```java\npublic class SarahApplication {\n    public static void main(String[] args) {\n        System.out.println(\"Sarah Ngin est active et opérationnelle.\");\n    }\n}\n```"
]

# ==============================================================================
# 2. CONSOLIDATION ET COMPILATION DU CORPUS
# ==============================================================================

def build_master_corpus(output_path: str = "data/processed/master_unified_corpus.txt") -> str:
    """Regroupe les textes littéraires et les données conversationnelles/techniques."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    with open(out_file, "w", encoding="utf-8") as f_out:
        # A. Injecter le corpus conversationnel répété (pondération forte)
        for _ in range(12):
            for conv in CONVERSATION_DATA:
                f_out.write(conv.strip() + "\n\n")
                total_lines += 2

        # B. Extraits choisis de littérature classique (structure grammaticale et lexicale)
        raw_dir = ROOT_DIR / "data" / "raw"
        if raw_dir.exists():
            for txt_file in raw_dir.glob("*.txt"):
                try:
                    count = 0
                    with open(txt_file, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            clean_line = line.strip()
                            if clean_line and len(clean_line) > 25:
                                f_out.write(clean_line + "\n")
                                total_lines += 1
                                count += 1
                                if count >= 300: # Échantillon de qualité pour équilibrer le corpus
                                    break
                except Exception as e:
                    print(f"Avertissement lecture {txt_file}: {e}", flush=True)

    print(f"[Corpus Maître] Compilé avec succès : {out_file} ({total_lines} lignes consolidées)", flush=True)
    return str(out_file)

# ==============================================================================
# 3. DATASET PYTORCH POUR FLUX CONTINU
# ==============================================================================

class MasterUnifiedDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int = 128, stride: int = 64):
        self.block_size = block_size
        self.stride = stride
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.indices = list(range(0, max(0, len(self.data) - self.block_size), self.stride))

    def __len__(self) -> int:
        return max(1, len(self.indices))

    def __getitem__(self, idx: int):
        if not self.indices:
            pad = torch.zeros(self.block_size, dtype=torch.long)
            return pad, pad
        start = self.indices[idx]
        chunk = self.data[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ==============================================================================
# 4. ENTRAÎNEMENT & COMPILATION NEURONALE PYTORCH
# ==============================================================================

def train_master_network(
    corpus_path: str,
    tokenizer_path: str = "checkpoints/sarah_tokenizer_8k.json",
    checkpoint_out: str = "checkpoints/sarah_ngin_master.pt",
    epochs: int = 20,
    batch_size: int = 16,
    block_size: int = 128,
    learning_rate: float = 5e-4,
    device_str: str = "cpu"
):
    print("=" * 70, flush=True)
    print("  ENTRAÎNEMENT DU RÉSEAU DE NEURONES MAÎTRE SARAH NGIN", flush=True)
    print("=" * 70, flush=True)

    device = torch.device(device_str)
    
    # 1. Charger le Tokenizer
    if not Path(tokenizer_path).exists():
        fallback_tok = "checkpoints/sarah_tokenizer.json"
        if Path(fallback_tok).exists():
            tokenizer_path = fallback_tok
    tokenizer = SarahTokenizer.load(tokenizer_path)
    print(f"[Tokenizer] Vocabulaire actif : {tokenizer.vocab_size} tokens", flush=True)

    # 2. Tokenisation complète du corpus
    print("[Données] Encodage du corpus unifié en tokens...", flush=True)
    all_tokens = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                all_tokens.extend(tokenizer.encode(s, add_bos=True, add_eos=True))

    print(f"[Données] Total tokens : {len(all_tokens):,}", flush=True)

    split_idx = int(len(all_tokens) * 0.90)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_ds = MasterUnifiedDataset(train_tokens, block_size=block_size, stride=32)
    val_ds = MasterUnifiedDataset(val_tokens, block_size=block_size, stride=64)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Configuration du Réseau Transformer (6 Couches, 6 Têtes, Dim 192, SwiGLU, RoPE)
    config = SarahNginConfig(
        vocab_size=tokenizer.vocab_size,
        dim=192,
        n_layers=6,
        n_heads=6,
        max_seq_len=256,
        dropout=0.05
    )

    model = SarahNginTransformer(config).to(device)
    params_info = model.count_parameters()
    print(f"[Architecture] Paramètres : {params_info['total_parameters']:,} (~{params_info['size_in_megabytes_fp32']:.2f} Mo FP32)", flush=True)

    # 4. Optimiseur & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=learning_rate * 0.1)

    best_val_loss = float("inf")
    loss_history = []

    # 5. Boucle d'époques PyTorch
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            steps += 1

        avg_train_loss = total_train_loss / max(1, steps)

        # Validation
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

        # Sauvegarde du checkpoint
        if avg_val_loss < best_val_loss or epoch == epochs:
            best_val_loss = avg_val_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "vocab_size": tokenizer.vocab_size
            }
            torch.save(ckpt_dict, checkpoint_out)
            torch.save(ckpt_dict, "checkpoints/sarah_ngin_best.pt")

        # Mise à jour du fichier de progression pour l'interface web
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
                "target_hardware": "Apple iPhone 14 (< 150 Mo RAM)",
                "last_log": f"Époque {epoch}/{epochs} validée (Loss: {avg_train_loss:.4f})",
                "elapsed_seconds": round(elapsed, 1)
            }, pf, indent=2)

    print("=" * 70, flush=True)
    print(f" Entraînement terminé avec succès ! Modèle maître sauvegardé : {checkpoint_out}", flush=True)
    print(f" Meilleure Loss de validation : {best_val_loss:.4f}", flush=True)
    print("=" * 70, flush=True)

    # 6. Test d'Inférence Neuronale Directe
    print("\n--- TEST D'INFÉRENCE NEURONALE DU MODÈLE MAÎTRE ---", flush=True)
    generator = SarahNginGenerator(model, tokenizer, device=str(device))
    test_prompts = [
        "Question: Bonjour Sarah, qui es-tu ?\nRéponse:",
        "Question: Quel est mon fruit préféré ?\nRéponse:",
        "Question: Écris une fonction Python pour additionner deux nombres.\nRéponse:"
    ]
    for p in test_prompts:
        gen = generator.generate(p, max_new_tokens=60, temperature=0.35, repetition_penalty=1.3)
        print(f"\nPrompt: {p}", flush=True)
        print(f"Génération Neuronale: {gen}", flush=True)

if __name__ == "__main__":
    corpus_file = build_master_corpus()
    train_master_network(
        corpus_path=corpus_file,
        tokenizer_path="checkpoints/sarah_tokenizer_8k.json",
        checkpoint_out="checkpoints/sarah_ngin_master.pt",
        epochs=20,
        batch_size=16,
        learning_rate=5e-4
    )
