"""
Entraînement Neuronal Intensif (Heavy Neural Training - 20 Époques).
Exécute de véritables calculs de gradients, rétropropagation et mises à jour de poids
sur le réseau de neurones Transformer Sarah Engine avec affichage direct de la perte (Loss).
"""

import sys
import time
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from model.transformer import SarahNginTransformer
from model.config import SarahNginConfig
from tokenizer.bpe_tokenizer import SarahTokenizer
from agent_developer.coding_corpus import generate_coding_corpus
from agent_developer.data_collector import DataCollector
from training.next_course_session import COURSE_3_MODULES
from training.advanced_course_session import COURSE_2_MODULES
from training.professor_supervisor_engine import EXAM_CURRICULUM

def run_intensive_training(num_epochs: int = 20, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  SARAH ENGINE : SESSION D'ENTRAÎNEMENT NEURONAL INTENSIF ({num_epochs} ÉPOQUES)")
    print(f"  Périphérique de calcul (Device) : {device.type.upper()}")
    print("=" * 80)

    # 1. Chargement du Tokenizer & Checkpoint
    tokenizer = SarahTokenizer.load("checkpoints/sarah_tokenizer.json")
    ckpt_path = Path("checkpoints/sarah_engine_code_trained.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/sarah_ngin_best.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = SarahNginTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture : {config.n_layers} couches | Dimension : {config.dim} | Têtes : {config.n_heads}")
    print(f"Paramètres totaux : {total_params:,} (Tous entraînables : {trainable_params:,})")

    # 2. Compilation de l'ensemble des données de code et connaissances
    print("\nCompilation du corpus d'entraînement multi-sources...")
    all_texts = []
    
    # Modules de code
    for c in [COURSE_2_MODULES, COURSE_3_MODULES]:
        for m in c:
            all_texts.append(f"<instruction>{m['objective']}</instruction>\n<code>{m['course_source']}</code>")
            
    for e in EXAM_CURRICULUM:
        all_texts.append(f"<instruction>{e['prompt']}</instruction>\n<code>{e['exemplar_solution']}</code>")

    coding_corpus = generate_coding_corpus()
    all_texts.extend(coding_corpus)

    # Dictionnaires et littérature
    collector = DataCollector()
    hebrew_french = collector.download_hebrew_french_massive_corpus()
    french_dict = collector.generate_french_dictionary_corpus()
    all_texts.extend(hebrew_french[:150])
    all_texts.extend(french_dict[:150])

    # Tokenisation et mise en forme des tenseurs
    tokenized_batches = []
    for txt in all_texts:
        toks = tokenizer.encode(txt, add_bos=True, add_eos=True)
        if len(toks) > config.max_seq_len:
            toks = toks[:config.max_seq_len]
        if len(toks) >= 8:
            tokenized_batches.append(torch.tensor(toks, dtype=torch.long, device=device))

    print(f"Nombre total de séquences d'entraînement : {len(tokenized_batches)}")

    # 3. Optimiseur AdamW avec Cosine Annealing Learning Rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    print("\n" + "-" * 80)
    print(f"{'Époque':<10} | {'Perte (Loss)':<15} | {'Perplexité':<15} | {'LR':<12} | {'Temps':<10}")
    print("-" * 80)

    model.train()
    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        step_count = 0

        # Permutation aléatoire pour varier les gradients
        indices = torch.randperm(len(tokenized_batches))

        for idx in indices:
            batch = tokenized_batches[idx]
            input_ids = batch[:-1].unsqueeze(0)
            targets = batch[1:].unsqueeze(0)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(step_count, 1)
        ppl = math.exp(min(avg_loss, 20))
        current_lr = scheduler.get_last_lr()[0]

        print(f"[{epoch:02d}/{num_epochs:02d}]     | {avg_loss:<15.4f} | {ppl:<15.2f} | {current_lr:<12.2e} | {elapsed:.2f}s")

    total_duration = time.time() - start_total_time
    print("-" * 80)
    print(f"[OK] Entrainement termine en {total_duration:.2f}s ({total_duration/60:.2f} min).")

    # 4. Sauvegarde des poids actualisés
    save_path = Path("checkpoints/sarah_engine_code_trained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": tokenizer.vocab_size,
        "identity": "Sarah Engine",
        "trained_epochs": num_epochs,
        "final_loss": avg_loss,
        "final_perplexity": ppl
    }, save_path)
    print(f"Poids neuronaux sauvegardés dans : {save_path}\n")

if __name__ == "__main__":
    run_intensive_training(num_epochs=20, lr=4e-4)
