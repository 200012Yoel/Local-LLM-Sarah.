"""
Entraînement Neuronal Intensif 30 Époques sur l'Architecture Réelle Transformer Sarah Engine.
Calcule de véritables gradients sur CPU/GPU avec affichage en direct dans la console.
"""

import sys
import time
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from model.transformer import SarahNginTransformer
from model.config import SarahNginConfig
from tokenizer.bpe_tokenizer import SarahTokenizer
from agent_developer.coding_corpus import generate_coding_corpus
from training.next_course_session import COURSE_3_MODULES
from training.advanced_course_session import COURSE_2_MODULES
from training.professor_supervisor_engine import EXAM_CURRICULUM

def train_real_transformer_30_epochs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DÉBUT DU VRAI ENTRAÎNEMENT NEURONAL LOURD (30 ÉPOQUES) ===")
    print(f"Périphérique : {device.type.upper()}")

    # 1. Chargement du tokenizer et du modèle Transformer réel
    tokenizer = SarahTokenizer.load("checkpoints/sarah_tokenizer.json")
    ckpt_path = Path("checkpoints/sarah_engine_code_trained.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/sarah_ngin_best.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = SarahNginTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    print(f"Modèle réel : Sarah Engine Transformer ({config.n_layers} couches, dim={config.dim}, vocab={tokenizer.vocab_size})")

    # 2. Préparation du corpus de code et instructions
    all_texts = []
    for c in [COURSE_2_MODULES, COURSE_3_MODULES]:
        for m in c:
            all_texts.append(f"<instruction>{m['objective']}</instruction>\n<code>{m['course_source']}</code>")
    for e in EXAM_CURRICULUM:
        all_texts.append(f"<instruction>{e['prompt']}</instruction>\n<code>{e['exemplar_solution']}</code>")
    all_texts.extend(generate_coding_corpus())

    # Tokenisation
    batches = []
    for txt in all_texts:
        toks = tokenizer.encode(txt, add_bos=True, add_eos=True)
        if len(toks) > config.max_seq_len:
            toks = toks[:config.max_seq_len]
        if len(toks) >= 8:
            batches.append(torch.tensor(toks, dtype=torch.long, device=device))

    print(f"Nombre de séquences d'entraînement : {len(batches)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)
    
    # 3. Boucle d'entraînement de 30 époques
    t0_global = time.time()
    for epoch in range(1, 31):
        epoch_loss = 0.0
        steps = 0
        t0_epoch = time.time()

        for b in batches:
            input_ids = b[:-1].unsqueeze(0)
            targets = b[1:].unsqueeze(0)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        ppl = math.exp(min(avg_loss, 20))
        epoch_dur = time.time() - t0_epoch
        print(f"Epoque [{epoch:02d}/30] - Loss calculee : {avg_loss:.4f} | Perplexite : {ppl:.2f} ({epoch_dur:.2f}s)", flush=True)

    total_time = time.time() - t0_global
    print(f"Duree totale d'entrainement : {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)

    # 4. Sauvegarde des poids
    save_path = Path("checkpoints/sarah_engine_code_trained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": tokenizer.vocab_size,
        "identity": "Sarah Engine",
        "trained_epochs": 30,
        "final_loss": avg_loss,
        "final_perplexity": ppl
    }, save_path)
    print("=== ENTRAINEMENT TERMINE : POIDS MIS A JOUR DANS checkpoints/sarah_engine_code_trained.pt ===", flush=True)

if __name__ == "__main__":
    train_real_transformer_30_epochs()
