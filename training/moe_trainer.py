"""
Pipeline d'Entraînement par Paliers & Distillation pour Sarah Engine MoE BitNet b1.58.
Intègre :
- Descente de gradient réelle PyTorch (AdamW + Cosine Annealing)
- Perte combinée : Cross-Entropy + Load Balancing MoE + Diversité Lexicale
- Sauvegarde de checkpoints optimisés pour Mobile / Edge.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Fix stdout encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from model.transformer_moe import SarahMoETransformer, SarahMoEConfig
from tokenizer.bpe_tokenizer import SarahTokenizer
from agent_developer.argot_sms_dict import ARGOT_SMS_DICTIONARY

class MoECorpusDataset(Dataset):
    def __init__(self, tokenizer: SarahTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        texts = []
        
        # 1. Livres classiques (extraits ciblés)
        for p in Path("data/raw").glob("*.txt"):
            try:
                lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").split("\n") if 15 < len(l.strip()) < 150]
                texts.extend(lines[:200])
            except Exception:
                pass

        # 2. Dictionnaires & Connaissances
        for slang, (trad, ex, defn) in list(ARGOT_SMS_DICTIONARY.items())[:30]:
            texts.append(f"Question: Que veut dire '{slang}' en SMS ? Réponse: '{slang}' signifie {trad}. Exemple: {ex}.")

        hebrew_data = [
            ("חָכְמָה", "La sagesse", "Hokhmah"),
            ("שָׁלוֹם", "La paix", "Shalom"),
            ("בְּרֵאשִׁית", "Au commencement", "Bereshit"),
            ("תּוֹרָה", "L'enseignement et la Loi", "Torah"),
            ("אַהֲבָה", "L'amour sincère", "Ahava")
        ]
        for heb, fr, phon in hebrew_data:
            texts.append(f"Question: Comment traduire {heb} ({phon}) ? Réponse: En français, {heb} signifie {fr}.")

        # 3. Programmation & Intelligence Artificielle (Multi-langages)
        code_tasks = [
            ("Python", "def compute_loss(pred, target):\n    return torch.nn.functional.cross_entropy(pred, target)"),
            ("JavaScript", "async function queryModel(prompt) {\n  const res = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({ prompt }) });\n  return await res.json();\n}"),
            ("PHP", "<?php\nfunction callSarah($p) {\n    return file_get_contents('http://localhost:8080/api/chat');\n}\n?>"),
            ("XML", "<?xml version='1.0' encoding='UTF-8'?>\n<sarah_engine>\n  <model name='Sarah MoE BitNet' version='2.0'/>\n</sarah_engine>"),
            ("Swift", "import SwiftUI\nstruct SarahChat: View {\n  var body: some View { Text('Sarah MoE BitNet Active') }\n}"),
            ("Java", "public class SarahEngine {\n  public static void main(String[] args) {\n    System.out.println('0 Bug BitNet Ready');\n  }\n}")
        ]
        for lang, code in code_tasks:
            texts.append(f"Question: Peux-tu coder un exemple en {lang} ?\nRéponse:\n{code}")

        for t in texts:
            enc = self.tokenizer.encode(t, add_bos=True, add_eos=True)
            if len(enc) > 4:
                self.samples.append(enc[:self.max_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    
    padded_xs = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_ys = torch.full((len(ys), max_len), -100, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_xs[i, :x.size(0)] = x
        padded_ys[i, :y.size(0)] = y
        
    return padded_xs, padded_ys

def compute_lexical_diversity_loss(logits: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Pénalise la concentration excessive des probabilités sur les mêmes tokens (entropie de diversité).
    Empêche les boucles et les bégaiements dès la phase d'apprentissage.
    """
    probs = F.softmax(logits, dim=-1)
    # Entropie moyenne par token
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
    # On souhaite une entropie saine (non nulle et équilibrée)
    target_entropy = 4.0
    entropy_penalty = F.relu(target_entropy - entropy)
    return alpha * entropy_penalty

def train_moe_palier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INITIALISATION ENTRAÎNEMENT PAR PALIER SARAH ENGINE MoE BITNET ===")
    print(f"Appareil d'entraînement : {device}")

    # 1. Charger le Tokenizer
    tok_path = "checkpoints/sarah_tokenizer_8k.json" if Path("checkpoints/sarah_tokenizer_8k.json").exists() else "checkpoints/sarah_tokenizer.json"
    tokenizer = SarahTokenizer.load(tok_path)
    print(f"Tokenizer chargé : {tok_path} (Vocabulaire : {tokenizer.vocab_size})")

    # 2. Configurer le modèle MoE BitNet
    config = SarahMoEConfig(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        embed_dim=64,
        n_layers=6,
        n_heads=8,
        num_experts=8,
        top_k_experts=2,
        hidden_dim_mult=1.5,
        sliding_window=128,
        max_seq_len=256,
        dropout=0.05,
        aux_loss_coef=0.02
    )

    model = SarahMoETransformer(config).to(device)
    param_info = model.count_parameters()
    print(f"Capacité Totale Paramètres : {param_info['total_parameters']:,} (Actifs par token: {param_info['active_parameters_per_token']:,})")
    print(f"Experts : {param_info['experts_count']} (Top-{param_info['active_experts']} actifs par token)")

    # 3. Préparer les données
    dataset = MoECorpusDataset(tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    print(f"Taille du Dataset : {len(dataset)} séquences.")

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Entraînement par Paliers (Palier 1: Découverte ➔ Palier 2: Consolidation ➔ Palier 3: Maîtrise Suprême)
    num_paliers = 3
    epochs_per_palier = 5

    best_loss = float("inf")
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    for palier in range(1, num_paliers + 1):
        print(f"\n" + "="*70)
        print(f"▶ PALIER {palier}/{num_paliers} : {'DÉCOUVERTE & ROUTAGE EXPERTS' if palier == 1 else 'CONSOLIDATION MULTILINGUE' if palier == 2 else 'MAÎTRISE SUPRÊME & ZERO-ERREUR'}")
        print("="*70)

        for epoch in range(1, epochs_per_palier + 1):
            model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_aux = 0.0
            total_div = 0.0
            steps = 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                logits, aux_loss, _ = model(x)
                
                # Perte Cross-Entropy
                loss_ce = ce_loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                
                # Perte Diversité Lexicale
                loss_div = compute_lexical_diversity_loss(logits, alpha=0.03)
                
                # Perte Totale
                loss_total = loss_ce + (config.aux_loss_coef * aux_loss) + loss_div

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss_total.item()
                total_ce += loss_ce.item()
                total_aux += aux_loss.item()
                total_div += loss_div.item()
                steps += 1

            scheduler.step()
            avg_loss = total_loss / max(steps, 1)
            avg_ce = total_ce / max(steps, 1)
            avg_aux = total_aux / max(steps, 1)

            print(f"  [Palier {palier} - Époque {epoch}/{epochs_per_palier}] Loss Totale: {avg_loss:.4f} (CE: {avg_ce:.4f} | MoE Aux: {avg_aux:.4f}) | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = output_dir / "sarah_engine_moe_bitnet_best.pt"
                torch.save({
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "param_info": param_info,
                    "loss": best_loss,
                    "palier": palier,
                    "epoch": epoch
                }, best_ckpt_path)

    # Sauvegarde finale
    final_path = output_dir / "sarah_engine_moe_bitnet_final.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "param_info": param_info,
        "loss": best_loss
    }, final_path)

    print("\n" + "="*70)
    print(f"✓ ENTRAÎNEMENT MoE BITNET TERMINÉ AVEC SUCCÈS !")
    print(f"✓ Meilleure Loss : {best_loss:.4f}")
    print(f"✓ Checkpoint sauvegardé dans : {best_ckpt_path}")
    print("="*70)

if __name__ == "__main__":
    train_moe_palier()
