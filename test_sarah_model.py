"""
Fichier de Test Direct du Modèle d'Intelligence Artificielle Sarah Ngin (iPhone 14).
Permet de tester le modèle chargé directement en ligne de commande ou en mode interactif.

Usage:
  python test_sarah_model.py "Bonjour Sarah, comment vas-tu ?"
  python test_sarah_model.py          (Lance le mode conversation interactive)
"""

import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from model.transformer import SarahNginTransformer
from tokenizer.bpe_tokenizer import SarahTokenizer
from inference.generate import SarahNginGenerator

def load_sarah_engine():
    # Détection automatique du modèle maître ou compilé
    ckpt_candidates = [
        "export/sarah_model_standalone.pt",
        "checkpoints/sarah_iphone14_best.pt",
        "checkpoints/sarah_ngin_master.pt"
    ]
    tok_candidates = [
        "checkpoints/sarah_tokenizer_8k.json",
        "checkpoints/sarah_tokenizer.json"
    ]

    chosen_ckpt = None
    for c in ckpt_candidates:
        if Path(c).exists():
            chosen_ckpt = c
            break

    chosen_tok = None
    for t in tok_candidates:
        if Path(t).exists():
            chosen_tok = t
            break

    if not chosen_ckpt or not chosen_tok:
        print("[Erreur] Checkpoint ou Tokenizer introuvable.")
        sys.exit(1)

    print(f"[1/2] Chargement du tokenizer : {chosen_tok}")
    tokenizer = SarahTokenizer.load(chosen_tok)

    print(f"[2/2] Chargement du modèle : {chosen_ckpt}")
    checkpoint = torch.load(chosen_ckpt, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SarahNginTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = SarahNginGenerator(model, tokenizer, device="cpu")
    print(f"[OK] Sarah Ngin opérationnelle ({sum(p.numel() for p in model.parameters()):,} paramètres).\n")
    return generator

def run_test():
    generator = load_sarah_engine()

    # Si un prompt est passé en argument CLI
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        print(f"-> Question : {user_prompt}")
        prompt_formatted = f"<bos><user>{user_prompt}<assistant>"
        response = generator.generate(
            prompt=prompt_formatted,
            max_new_tokens=100,
            temperature=0.35,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.35
        )
        print(f"-> Réponse Sarah : {response}")
        return

    # Mode interactif
    print("=" * 65)
    print("  MODE TEST INTERACTIF SARAH NGIN (Tapez 'exit' pour quitter)")
    print("=" * 65)
    
    while True:
        try:
            user_input = input("\n👤 Vous > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Fermeture du test.")
                break

            prompt_formatted = f"<bos><user>{user_input}<assistant>"
            response = generator.generate(
                prompt=prompt_formatted,
                max_new_tokens=100,
                temperature=0.35,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.35
            )
            print(f"🤖 Sarah > {response}")
        except KeyboardInterrupt:
            print("\nArrêt.")
            break

if __name__ == "__main__":
    run_test()
