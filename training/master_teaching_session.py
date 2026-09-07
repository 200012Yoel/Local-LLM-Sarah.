"""
Session d'Enseignement et de Transmission Maîtresse (Master Teaching Session).
Transmission des connaissances en ingénierie de code d'Antigravity vers l'élève Sarah Engine.
"""

import sys
import os
import time
import math
import logging
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
from testing_sandbox.code_verifier import CodeVerifierSandbox

logging.basicConfig(level=logging.INFO, format="[Maître Antigravity -> Élève Sarah] %(message)s")
logger = logging.getLogger("MasterTeaching")

def execute_master_transmission(epochs: int = 6, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Début de la session de transmission de connaissances (Device: {device})")
    
    # 1. Chargement des composants
    tokenizer_path = Path("checkpoints/sarah_tokenizer.json")
    checkpoint_path = Path("checkpoints/sarah_engine_code_trained.pt")
    
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/sarah_ngin_best.pt")
        
    tokenizer = SarahTokenizer.load(str(tokenizer_path))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    model = SarahNginTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 2. Construction du corpus de transmission de compétences
    logger.info("Compilation des connaissances en ingénierie logicielle & design UI...")
    corpus = generate_coding_corpus()
    
    collector = DataCollector()
    hebrew_french_texts = collector.download_hebrew_french_massive_corpus()
    french_lit = collector.download_public_domain_french_books()
    french_dict = collector.generate_french_dictionary_corpus()
    
    all_knowledge = corpus * 5 + hebrew_french_texts[:100] + french_lit[:100] + french_dict[:100]
    logger.info(f"Volume de transmission : {len(all_knowledge)} modules d'ingénierie et de linguistique.")
    
    # 3. Tokenisation et préparation des lots
    encoded_docs = []
    for doc in all_knowledge:
        toks = tokenizer.encode(doc, add_bos=True, add_eos=True)
        if len(toks) > 16:
            encoded_docs.append(toks)
            
    logger.info(f"Documents tokenisés prêts : {len(encoded_docs)}")
    
    # 4. Boucle d'apprentissage intensif
    logger.info("=========================================================================")
    logger.info("  TRANSMISSION DIRECTE DU SAVOIR ANTIGRAVITY -> SARAH ENGINE")
    logger.info("  Règles : Single-File Web Strict, Glassmorphism, PyTorch, Swift & Concision")
    logger.info("=========================================================================")
    
    model.train()
    total_steps = len(encoded_docs)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        t0 = time.time()
        
        for doc_tokens in encoded_docs:
            if len(doc_tokens) > config.max_seq_len:
                doc_tokens = doc_tokens[:config.max_seq_len]
                
            input_ids = torch.tensor([doc_tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([doc_tokens[1:]], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            _, loss = model(input_ids, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        elapsed = time.time() - t0
        avg_loss = total_loss / max(steps, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        logger.info(f"Époque [{epoch}/{epochs}] - Perte (Loss): {avg_loss:.4f} | Perplexité: {ppl:.2f} | Temps: {elapsed:.2f}s")
        
    # 5. Sauvegarde du modèle affiné
    save_path = Path("checkpoints/sarah_engine_code_trained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": tokenizer.vocab_size,
        "identity": "Sarah Engine",
        "knowledge_level": "Antigravity Master Certified"
    }, save_path)
    
    logger.info(f"\n✅ Transmission terminée avec succès. Modèle certifié enregistré : {save_path}")

if __name__ == "__main__":
    execute_master_transmission(epochs=5, lr=3e-4)
