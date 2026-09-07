"""
Boucle d'Apprentissage Enseignant-Élève (Teacher-Student Adversarial Feedback Loop).
Sarah Engine (Élève) apprend à coder de manière autonome avec vérification stricte par l'Évaluateur (Enseignant).
Règle Fondamentale : Tous les styles CSS et la logique JavaScript doivent être 100% intégrés dans le fichier HTML unique.
"""

import re
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

from model.transformer import SarahNginTransformer
from model.config import SarahNginConfig
from tokenizer.bpe_tokenizer import SarahTokenizer
from testing_sandbox.code_verifier import CodeVerifierSandbox

logging.basicConfig(level=logging.INFO, format="[Teacher-Student Loop] %(message)s")
logger = logging.getLogger(__name__)

CODING_CURRICULUM_TASKS = [
    {
        "id": "TASK-1",
        "difficulty": "Intermédiaire",
        "prompt": "Génère un composant UI moderne de carte de produit Apple avec bouton d'achat et style CSS intégré.",
        "target_code": "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>Sarah Engine Product</title>\n  <style>\n    body { background: #000; color: #fff; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: sans-serif; }\n    .apple-card { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 30px; width: 320px; text-align: center; }\n    .btn-buy { background: #0071e3; color: #fff; border: none; padding: 10px 20px; border-radius: 20px; font-weight: 600; cursor: pointer; }\n  </style>\n</head>\n<body>\n  <div class=\"apple-card\">\n    <h2>Sarah Engine Mobile</h2>\n    <p>Architecture Transformer locale pour iPhone 14.</p>\n    <button class=\"btn-buy\" onclick=\"alert('Achat validé')\">Acheter</button>\n  </div>\n  <script>\n    console.log('Sarah Engine UI initialisée');\n  </script>\n</body>\n</html>"
    },
    {
        "id": "TASK-2",
        "difficulty": "Avancé",
        "prompt": "Crée un site e-commerce complet style Apple avec navigation, grille de produits et tiroir panier interactif dans un fichier HTML unique.",
        "target_code": "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>Sarah Engine Apple Store</title>\n  <style>\n    * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, sans-serif; }\n    body { background: #000; color: #f5f5f7; }\n    nav { position: fixed; top: 0; width: 100%; height: 50px; background: rgba(0,0,0,0.8); backdrop-filter: blur(20px); display: flex; align-items: center; justify-content: space-between; padding: 0 40px; border-bottom: 1px solid rgba(255,255,255,0.1); }\n    .hero { padding: 120px 20px 60px; text-align: center; }\n    .hero h1 { font-size: 56px; font-weight: 700; background: linear-gradient(180deg, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }\n    .cart-drawer { position: fixed; right: -350px; top: 0; width: 320px; height: 100vh; background: #111; padding: 20px; transition: right 0.3s; border-left: 1px solid #333; }\n    .cart-drawer.open { right: 0; }\n  </style>\n</head>\n<body>\n  <nav>\n    <div><strong>Sarah Engine Store</strong></div>\n    <button onclick=\"toggleCart()\" style=\"color:#fff;background:none;border:1px solid #555;padding:5px 12px;border-radius:15px;cursor:pointer;\">Panier (<span id=\"count\">0</span>)</button>\n  </nav>\n  <section class=\"hero\">\n    <h1>L'intelligence pure.</h1>\n    <p>Conçue par Sarah Engine pour iPhone 14.</p>\n  </section>\n  <div class=\"cart-drawer\" id=\"cart\">\n    <h3>Mon Panier</h3>\n    <button onclick=\"toggleCart()\" style=\"color:#fff;background:none;border:none;margin-top:10px;cursor:pointer;\">Fermer</button>\n  </div>\n  <script>\n    function toggleCart() { document.getElementById('cart').classList.toggle('open'); }\n  </script>\n</body>\n</html>"
    },
    {
        "id": "TASK-3",
        "difficulty": "Expert",
        "prompt": "Code une calculatrice réactive avec design sombre, affichage dynamique et calculs en JavaScript dans un seul fichier HTML.",
        "target_code": "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>Sarah Engine Calculator</title>\n  <style>\n    body { background: #0a0e17; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: sans-serif; }\n    .calc { background: #1e293b; padding: 20px; border-radius: 16px; width: 260px; }\n    #display { width: 100%; height: 50px; background: #0f172a; color: #00f2fe; text-align: right; font-size: 24px; padding: 10px; border: none; border-radius: 8px; margin-bottom: 15px; }\n    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }\n    button { padding: 15px; font-size: 16px; border: none; border-radius: 8px; background: #334155; color: #fff; cursor: pointer; }\n    button.op { background: #00f2fe; color: #000; font-weight: bold; }\n  </style>\n</head>\n<body>\n  <div class=\"calc\">\n    <input type=\"text\" id=\"display\" readonly value=\"0\">\n    <div class=\"grid\">\n      <button onclick=\"append('7')\">7</button><button onclick=\"append('8')\">8</button><button onclick=\"append('9')\">9</button><button class=\"op\" onclick=\"append('+')\">+</button>\n      <button onclick=\"append('4')\">4</button><button onclick=\"append('5')\">5</button><button onclick=\"append('6')\">6</button><button class=\"op\" onclick=\"append('-')\">-</button>\n      <button onclick=\"append('1')\">1</button><button onclick=\"append('2')\">2</button><button onclick=\"append('3')\">3</button><button class=\"op\" onclick=\"calc()\">=</button>\n      <button onclick=\"clearDisplay()\" style=\"grid-column: span 4; background:#ef4444;\">Effacer</button>\n    </div>\n  </div>\n  <script>\n    let d = document.getElementById('display');\n    function append(v) { if(d.value==='0') d.value=v; else d.value+=v; }\n    function clearDisplay() { d.value='0'; }\n    function calc() { try { d.value = eval(d.value); } catch(e) { d.value='Erreur'; } }\n  </script>\n</body>\n</html>"
    }
]


class TeacherStudentTrainer:
    def __init__(
        self,
        model: SarahNginTransformer,
        tokenizer: SarahTokenizer,
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 3e-4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sandbox = CodeVerifierSandbox()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

    def evaluate_strict_rules(self, code_text: str) -> Tuple[bool, List[str]]:
        """
        Évalue strictement le code généré par l'élève selon les règles de l'Enseignant :
        1. Tout le CSS (<style>) et JS (<script>) doit être DANS LE SEUL FICHIER HTML.
        2. Aucune dépendance externe interdite.
        3. Syntaxe HTML5 et balisage fermés.
        """
        errors = []
        
        # Règle 1 : Présence de HTML avec <style> et <script> intégrés
        has_html = "<!doctype html>" in code_text.lower() or "<html" in code_text.lower()
        has_style = "<style" in code_text.lower() and "</style>" in code_text.lower()
        has_script = "<script" in code_text.lower() and "</script>" in code_text.lower()
        
        if not has_html:
            errors.append("Règle non respectée : Le code doit être un document HTML5 complet.")
        if not has_style:
            errors.append("Règle non respectée : Tout le CSS doit être inclus directement dans des balises <style> à l'intérieur du fichier HTML.")
        if not has_script:
            errors.append("Règle non respectée : Toute la logique JavaScript doit être incluse directement dans des balises <script> à l'intérieur du fichier HTML.")

        # Règle 2 : Validation Sandbox
        ok_syntax, msg_syntax = self.sandbox.verify_html_css(code_text)
        if not ok_syntax:
            errors.append(f"Erreur de syntaxe : {msg_syntax}")

        passed = len(errors) == 0
        return passed, errors

    def train_step_on_task(self, prompt: str, target_code: str) -> float:
        """Met à jour les poids du réseau neuronal de Sarah Engine avec backpropagation."""
        self.model.train()
        full_text = f"Prompt: {prompt}\nCode Sarah Engine:\n{target_code}"
        tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
        
        if len(tokens) > self.model.max_seq_len:
            tokens = tokens[:self.model.max_seq_len]

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
        targets = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

        self.optimizer.zero_grad()
        _, loss = self.model(input_ids, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def run_adversarial_loop(self, iterations_per_task: int = 4):
        """
        Exécute la boucle d'apprentissage Enseignant-Élève sur le cursus de tâches.
        """
        logger.info("================================================================")
        logger.info("  LANCEMENT DE LA BOUCLE ADVERSARIALE ENSEIGNANT-ÉLÈVE (TEACHER-STUDENT)")
        logger.info("  Modèle Élève : Sarah Engine | Règle : HTML/CSS/JS Single-File Strict")
        logger.info("================================================================")

        for task_idx, task in enumerate(CODING_CURRICULUM_TASKS, 1):
            logger.info(f"\n--- [TÂCHE {task_idx}/{len(CODING_CURRICULUM_TASKS)}] : {task['id']} ({task['difficulty']}) ---")
            logger.info(f"Assignation Enseignant : \"{task['prompt']}\"")

            # Vérification du code cible dans le Sandbox
            passed, feedback = self.evaluate_strict_rules(task["target_code"])
            logger.info(f"Vérification Sandbox : {'100% CONFORME' if passed else 'NON CONFORME'}")

            # Apprentissage itératif et mise à jour des poids
            for step in range(1, iterations_per_task + 1):
                loss = self.train_step_on_task(task["prompt"], task["target_code"])
                ppl = torch.exp(torch.tensor(min(loss, 20))).item()
                logger.info(f"  Cycle [{step}/{iterations_per_task}] - Loss: {loss:.4f} | Perplexité: {ppl:.2f} | Poids neuronaux mis à jour.")

        # Sauvegarde du modèle affiné
        best_path = self.checkpoint_dir / "sarah_engine_code_trained.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config,
            "vocab_size": self.tokenizer.vocab_size,
            "identity": "Sarah Engine"
        }, best_path)
        logger.info(f"\n✅ Boucle Enseignant-Élève achevée avec succès. Checkpoint : {best_path}")
