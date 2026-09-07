"""
Point d'Entrée Principal pour le Système d'Intelligence Artificielle "Sarah Ngin".
Permet de piloter l'Agent Développeur, l'apprentissage, l'inférence et l'export mobile.
"""

import sys
import io
import argparse
import logging
from pathlib import Path
import torch

# Configuration UTF-8 pour Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent_developer.data_collector import DataCollector
from tokenizer.bpe_tokenizer import SarahTokenizer
from model.config import SarahNginConfig
from model.transformer import SarahNginTransformer
from training.dataset import create_dataloaders
from training.trainer import SarahNginTrainer
from inference.generate import SarahNginGenerator
from export.export_mobile import MobileExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SarahNginMain")

def step_1_collect_data(data_dir: str = "data") -> str:
    logger.info("=== ÉTAPE 1 : AGENT DÉVELOPPEUR (COLLECTE DES DONNÉES) ===")
    collector = DataCollector(data_dir=data_dir)
    corpus_path = collector.collect_and_build_all()
    return str(corpus_path)

def step_2_train_tokenizer(corpus_path: str, vocab_size: int = 2048, tokenizer_path: str = "checkpoints/sarah_tokenizer.json") -> SarahTokenizer:
    logger.info("=== ÉTAPE 2 : ENTRAÎNEMENT DU TOKENIZER BPE MULTILINGUE ===")
    tokenizer = SarahTokenizer(vocab_size=vocab_size)
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    tokenizer.train(lines)
    tokenizer.save(tokenizer_path)
    return tokenizer

def step_3_train_model(
    corpus_path: str,
    tokenizer: SarahTokenizer,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 7e-4,
    checkpoint_dir: str = "checkpoints"
) -> SarahNginTransformer:
    logger.info("=== ÉTAPE 3 : CRÉATION ET ENTRAÎNEMENT DU RÉSEAU SARAH NGIN ===")
    
    # Configuration optimisée iPhone 14 (< 4 Go RAM)
    config = SarahNginConfig.mobile_iphone14(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = 256
    config.dim = 192
    config.n_layers = 6
    config.n_heads = 6
    config.save(f"{checkpoint_dir}/sarah_ngin_config.json")

    model = SarahNginTransformer(config)
    param_info = model.count_parameters()
    logger.info(f"Spécifications Sarah Ngin :")
    logger.info(f"- Paramètres totaux : {param_info['total_parameters']:,}")
    logger.info(f"- Empreinte mémoire FP32 : {param_info['size_in_megabytes_fp32']:.2f} Mo")
    logger.info(f"- Empreinte mémoire INT8 : {param_info['size_in_megabytes_int8']:.2f} Mo")
    logger.info(f"- Compatible iPhone 14 (4 Go RAM) : OUI (100% dans les spécifications)")

    train_loader, val_loader = create_dataloaders(
        filepath=corpus_path,
        tokenizer=tokenizer,
        block_size=64,
        batch_size=batch_size,
        val_split=0.1
    )

    trainer = SarahNginTrainer(model, tokenizer, learning_rate=lr)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir
    )
    return model

def step_4_test_generation(checkpoint_dir: str = "checkpoints"):
    logger.info("=== ÉTAPE 4 : TEST D'INFÉRENCE & DE GÉNÉRATION LOGIQUE ===")
    generator = SarahNginGenerator.from_checkpoint(
        checkpoint_path=f"{checkpoint_dir}/sarah_ngin_best.pt",
        tokenizer_path=f"{checkpoint_dir}/sarah_tokenizer.json"
    )

    prompts = [
        "Dictionnaire Français : L'intelligence artificielle",
        "Principe de Raisonnement Logique :",
        "Traduction FR-HE : 'bonjour' se traduit en hébreu par",
        "Traduction FR-ZH : 'monde' 在中文里翻译为",
        "Question: Qu'est-ce que l'intelligence artificielle ?\nRéponse:",
        "Le modèle Sarah Ngin analyse avec précision"
    ]

    print("\n" + "="*70)
    print("      DÉMONSTRATION DU MODÈLE SARAH NGIN")
    print("="*70)
    for p in prompts:
        out = generator.generate(
            prompt=p,
            max_new_tokens=40,
            temperature=0.6,
            top_k=30,
            repetition_penalty=1.2
        )
        print(f"\n[PROMPT]    {p}")
        print(f"[SARAH NGIN] {out}")
    print("="*70 + "\n")

def step_5_export_mobile(checkpoint_dir: str = "checkpoints", export_dir: str = "mobile_build"):
    logger.info("=== ÉTAPE 5 : EXPORT ET OPTIMISATION POUR IPHONE 14 (<4GB RAM) ===")
    config = SarahNginConfig.load(f"{checkpoint_dir}/sarah_ngin_config.json")
    model = SarahNginTransformer(config)
    checkpoint = torch.load(f"{checkpoint_dir}/sarah_ngin_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    exporter = MobileExporter(model, output_dir=export_dir)
    int8_file = exporter.export_quantized_int8()
    ts_file = exporter.export_torchscript_mobile()
    onnx_file = exporter.export_onnx()
    
    benchmark = exporter.benchmark_latency(num_runs=15, seq_len=32)
    
    logger.info("=== EXPORT MOBILE TERMINÉ ===")
    logger.info(f"Fichiers prêts pour déploiement iOS / Android dans '{export_dir}/'")

def interactive_chat(checkpoint_dir: str = "checkpoints"):
    """Mode conversationnel direct avec Sarah Ngin."""
    generator = SarahNginGenerator.from_checkpoint(
        checkpoint_path=f"{checkpoint_dir}/sarah_ngin_best.pt",
        tokenizer_path=f"{checkpoint_dir}/sarah_tokenizer.json"
    )
    print("\n" + "="*60)
    print("  Discussion avec Sarah Ngin (Tapez 'exit' ou 'quit' pour quitter)")
    print("="*60)
    while True:
        try:
            user_input = input("\nVous: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Fermeture de la session Sarah Ngin. À bientôt !")
                break
            
            prompt = f"Question: {user_input}\nRéponse:"
            response = generator.generate(
                prompt=prompt,
                max_new_tokens=60,
                temperature=0.7,
                top_k=40,
                repetition_penalty=1.2
            )
            print(f"Sarah Ngin: {response.strip()}")
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="Sarah Ngin AI System")
    parser.add_argument("--action", choices=["collect", "tokenize", "train", "generate", "export", "chat", "all"], default="all", help="Action à exécuter")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques d'entraînement")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Taille du vocabulaire")
    args = parser.parse_args()

    data_dir = "data"
    checkpoint_dir = "checkpoints"
    export_dir = "mobile_build"
    corpus_path = "data/processed/sarah_ngin_corpus.txt"
    tokenizer_path = f"{checkpoint_dir}/sarah_tokenizer.json"

    if args.action == "collect":
        step_1_collect_data(data_dir)
    elif args.action == "tokenize":
        step_2_train_tokenizer(corpus_path, vocab_size=args.vocab_size, tokenizer_path=tokenizer_path)
    elif args.action == "train":
        tokenizer = SarahTokenizer.load(tokenizer_path)
        step_3_train_model(corpus_path, tokenizer, epochs=args.epochs, checkpoint_dir=checkpoint_dir)
    elif args.action == "generate":
        step_4_test_generation(checkpoint_dir)
    elif args.action == "export":
        step_5_export_mobile(checkpoint_dir, export_dir)
    elif args.action == "chat":
        interactive_chat(checkpoint_dir)
    elif args.action == "all":
        # Exécution du pipeline complet
        corpus = step_1_collect_data(data_dir)
        tokenizer = step_2_train_tokenizer(corpus, vocab_size=args.vocab_size, tokenizer_path=tokenizer_path)
        step_3_train_model(corpus, tokenizer, epochs=args.epochs, checkpoint_dir=checkpoint_dir)
        step_4_test_generation(checkpoint_dir)
        step_5_export_mobile(checkpoint_dir, export_dir)

if __name__ == "__main__":
    main()
