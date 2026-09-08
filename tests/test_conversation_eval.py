import sys
import os
import json
from pathlib import Path

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from inference.generate import SarahNginGenerator

def run_conversation_evaluation(
    checkpoint_path: str = "checkpoints/sarah_iphone14_best.pt",
    tokenizer_path: str = "checkpoints/sarah_tokenizer_8k.json"
):
    print("=" * 75)
    print("  ÉVALUATION CONVERSATIONNELLE AUTOMATIQUE DU MODÈLE SARAH NGIN")
    print("=" * 75)

    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/sarah_ngin_master.pt"

    generator = SarahNginGenerator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path
    )

    test_battery = [
        # 1. Salutations et Présentation (FR & EN)
        {"category": "Identité & Accueil (FR)", "prompt": "<bos><user>Bonjour Sarah, comment vas-tu aujourd'hui ?<assistant>"},
        {"category": "Identity & Greeting (EN)", "prompt": "<bos><user>Hello! Who are you?<assistant>"},
        
        # 2. Connaissances Générales & Sciences
        {"category": "Sciences", "prompt": "<bos><user>Pourquoi le ciel est bleu ?<assistant>"},
        {"category": "Géographie", "prompt": "<bos><user>Quelle est la capitale de la France ?<assistant>"},
        {"category": "General Knowledge (EN)", "prompt": "<bos><user>What is a computer?<assistant>"},
        
        # 3. Informatique & Programmation
        {"category": "Code Python", "prompt": "<bos><user>Qu'est-ce que Python et à quoi ça sert ?<assistant>"},
        {"category": "Programming (EN)", "prompt": "<bos><user>Explain what a programming language is.<assistant>"},
        
        # 4. Mathématiques & Raisonnement
        {"category": "Calcul Arithmétique", "prompt": "<bos><user>Combien font 12 + 15 ?<assistant>"},
        {"category": "Arithmetic (EN)", "prompt": "<bos><user>What is 15 plus 25?<assistant>"},
        
        # 5. Traduction
        {"category": "Traduction FR -> EN", "prompt": "<bos><user>Traduis 'Le savoir est une force' en anglais.<assistant>"},
        {"category": "Translation EN -> FR", "prompt": "<bos><user>Translate 'good morning' into French.<assistant>"},
        
        # 6. Multi-Tours avec Contexte
        {"category": "Multi-Tour (Rétention Nom)", "prompt": "<bos><user>Je m'appelle Paul.<assistant>Enchanté Paul ! Comment puis-je t'aider ?<user>Quel est mon prénom ?<assistant>"},
        {"category": "Multi-Tour (Rétention Fruit)", "prompt": "<bos><user>Mon fruit préféré est la papaye.<assistant>C'est bien noté ! La papaye est un fruit tropical délicieux.<user>Quel est mon fruit préféré ?<assistant>"}
    ]

    results = []
    for test in test_battery:
        res = generator.generate(
            prompt=test["prompt"],
            max_new_tokens=80,
            temperature=0.35,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.3
        )
        print(f"\n[{test['category']}]")
        print(f"Prompt : {test['prompt']}")
        print(f"Sarah Ngin -> {res}")
        results.append({
            "category": test["category"],
            "prompt": test["prompt"],
            "response": res
        })

    with open("tests/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(" Évaluation terminée avec succès ! Résultats enregistrés dans tests/eval_results.json")
    print("=" * 75)

if __name__ == "__main__":
    run_conversation_evaluation()
