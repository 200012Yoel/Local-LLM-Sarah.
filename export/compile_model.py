"""
Script de compilation et exportation JIT / TorchScript autonome pour Sarah Ngin iPhone 14.
Génère un artefact de modèle compilé autonome prêt pour inférence directe.
"""

import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.transformer import SarahNginTransformer
from tokenizer.bpe_tokenizer import SarahTokenizer

def compile_sarah():
    print("=== COMPILATION DU MODÈLE SARAH NGIN IPHONE 14 ===")
    ckpt_path = "checkpoints/sarah_iphone14_best.pt"
    tok_path = "checkpoints/sarah_tokenizer_8k.json"
    out_dir = Path("export")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SarahTokenizer.load(tok_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SarahNginTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Sauvegarde d'un package compilé autonome (Modèle + Config + Vocabulaire + Weights)
    compiled_package = {
        "model_name": "Sarah Ngin iPhone 14 Master",
        "architecture": "Decoder-only Transformer with RoPE & SwiGLU & GQA",
        "parameters": 18556416,
        "config": config,
        "vocab_size": tokenizer.vocab_size,
        "model_state_dict": model.state_dict(),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "target_hardware": "Apple iPhone 14 (A15 Bionic)"
    }
    
    compiled_path = out_dir / "sarah_model_standalone.pt"
    torch.save(compiled_package, compiled_path)
    print(f"-> Modèle compilé et packagé sauvegardé : {compiled_path} ({compiled_path.stat().st_size / (1024*1024):.2f} Mo)")

    # TorchScript Traced Module pour inférence C++ / iOS
    dummy_input = torch.randint(0, config.vocab_size, (1, 16), dtype=torch.long)
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        jit_path = out_dir / "sarah_torchscript.pt"
        traced_model.save(str(jit_path))
        print(f"-> Modèle TorchScript JIT compilé : {jit_path} ({jit_path.stat().st_size / (1024*1024):.2f} Mo)")
    except Exception as e:
        print(f"[TorchScript Note] {e}")

    return compiled_path

if __name__ == "__main__":
    compile_sarah()
