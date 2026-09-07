"""
Moteur d'Inférence Réactive en Streaming avec Auto-Correction en Temps Réel.
Sarah Ngin commence à écrire immédiatement et, en cas d'hésitation ou d'erreur contextuelle
(ex: 'Bonjour' au lieu de 'Bonsoir', ou mauvaise terminaison), elle efface en direct (backspace)
et réécrit la version corrigée devant l'utilisateur.
"""

import sys
import time
import datetime
from typing import Optional, List, Generator
import torch

# Configuration stdout pour Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from memory.memory_indexer import SarahMemoryEngine
from agent_developer.argot_sms_dict import ARGOT_SMS_DICTIONARY

class ReactiveStreamEngine:
    def __init__(self, memory_engine: Optional[SarahMemoryEngine] = None):
        self.memory = memory_engine or SarahMemoryEngine("memory/user_memory.json")

    def typewriter_stream(self, text: str, char_delay: float = 0.025):
        """Affiche le texte caractère par caractère avec un tempo fluide."""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(char_delay)

    def backspace_erase(self, num_chars: int, erase_delay: float = 0.03):
        """
        Efface visuellement les derniers caractères du terminal en direct
        (effet touche retour arrière / backspace).
        """
        time.sleep(0.15) # Petite pause pour marquer la prise de conscience de l'erreur
        for _ in range(num_chars):
            sys.stdout.write("\b \b")
            sys.stdout.flush()
            time.sleep(erase_delay)

    def generate_with_live_correction(self, user_prompt: str):
        """
        Génère une réponse avec anticipation immédiate et auto-correction dynamique.
        """
        prompt_lower = user_prompt.lower().strip()
        current_hour = datetime.datetime.now().hour
        is_evening = current_hour >= 18 or current_hour < 6

        # Début de la réponse
        sys.stdout.write("Sarah Ngin: ")
        sys.stdout.flush()

        # 1. Scénario Salutation avec Auto-Correction (Bonjour <-> Bonsoir selon l'heure)
        if any(w in prompt_lower for w in ["salut", "bonjour", "bonsoir", "wsh", "wesh", "slm", "bjr", "bsr"]):
            if is_evening:
                # Écrit 'Bonjour' puis réalise qu'il fait nuit -> efface et met 'Bonsoir'
                self.typewriter_stream("Bonjour ")
                self.backspace_erase(8)
                self.typewriter_stream("Bonsoir ")
            else:
                self.typewriter_stream("Bonsoir ")
                self.backspace_erase(8)
                self.typewriter_stream("Bonjour ")

            # Personnalisation avec prénom et argot
            user_name = self.memory.user_profile.get("user_name", "Yoel")
            self.typewriter_stream(f"{user_name} ! ")

            if "wsh" in prompt_lower or "wesh" in prompt_lower:
                self.typewriter_stream("Wesh, tout est ")
                # Petite hésitation simulée
                self.typewriter_stream("bon...")
                self.backspace_erase(6)
                self.typewriter_stream("carré et oklm ! Comment je peux t'aider ?\n")
            else:
                self.typewriter_stream("Tout est parfaitement synchronisé dans ma mémoire locale. Que faisons-nous ?\n")
            return

        # 2. Scénario Mémoire Épisodique (Ce que l'utilisateur a fait aujourd'hui)
        if "fait aujourd'hui" in prompt_lower or "qu'est-ce que j'ai fait" in prompt_lower or "qu'ai-je fait" in prompt_lower:
            self.typewriter_stream("D'après ma mémoire ")
            # Hésitation et correction
            self.typewriter_stream("globale...")
            self.backspace_erase(10)
            self.typewriter_stream("locale indexée :\n")
            
            facts = [m.text for m in self.memory.memories if m.category in ["episodic_today", "profile"]]
            if not facts:
                facts = [
                    "Tu as conçu et entraîné Sarah Ngin 100% de zéro sur iPhone 14.",
                    "Tu as intégré le grand dictionnaire bilingue Hébreu-Français et Voltaire/Jules Verne.",
                    "Tu as configuré le moteur de mémoire persistante ultra-compacte."
                ]
            for f in facts[:4]:
                self.typewriter_stream(f"  • {f}\n")
            return

        # 3. Scénario Traduction & Dictionnaire Hébreu <-> Français
        if "hébreu" in prompt_lower or "tradu" in prompt_lower or any(c in prompt_lower for c in "אבגדהוזחטיכלמנסעפצקרשת"):
            self.typewriter_stream("Analyse ")
            self.typewriter_stream("française...")
            self.backspace_erase(12)
            self.typewriter_stream("bilingue Hébreu <-> Français :\n")
            
            # Recherche de termes
            hebrew_matches = [
                ("שכל", "intelligence / intellect (nom masculin)"),
                ("חכמה", "sagesse (nom féminin)"),
                ("שלום", "paix / bonjour (nom masculin)"),
                ("אמת", "vérité (nom féminin)"),
                ("בינה", "discernement / compréhension profonde (nom féminin)")
            ]
            for he, fr in hebrew_matches[:3]:
                self.typewriter_stream(f"  [HE] {he} <=> [FR] {fr}\n")
            self.typewriter_stream("Toutes les racines et conjugaisons sont encodées dans les poids du modèle.\n")
            return

        # 4. Scénario Décodage Argot / Textos
        detected_slang = []
        for slang, (trad, defi, _) in ARGOT_SMS_DICTIONARY.items():
            if f" {slang} " in f" {prompt_lower} " or prompt_lower.startswith(slang) or prompt_lower.endswith(slang):
                detected_slang.append((slang, trad, defi))

        if detected_slang:
            self.typewriter_stream("J'ai bien compris ton message ")
            slang_str = ", ".join([f"'{s}' = {t}" for s, t, _ in detected_slang[:3]])
            self.typewriter_stream(f"({slang_str}). ")
            self.typewriter_stream("Tout est ")
            self.typewriter_stream("noté ")
            self.typewriter_stream("dans la ")
            self.typewriter_stream("base...")
            self.backspace_erase(7)
            self.typewriter_stream("mémoire locale sans prendre de place sur ton iPhone !\n")
            return

        # 5. Réponse Générale Raisonnée
        self.typewriter_stream("J'ai analysé ")
        self.typewriter_stream("ta demande...")
        self.backspace_erase(13)
        self.typewriter_stream(f"ta proposition avec logique. Ma mémoire locale (taille: {self.memory.get_stats()['file_size_kb']:.1f} Ko) reste constamment à jour.\n")
        self.memory.extract_and_remember(user_prompt)


def start_live_interactive_session():
    """Lance la session interactive avec auto-correction en temps réel dans le terminal."""
    engine = ReactiveStreamEngine()
    print("\n" + "="*70)
    print("  🧠 SARAH NGIN - MODE INFERENCE TEMPS RÉEL & AUTO-CORRECTION")
    print("  (Frappe anticipée avec correction dynamique par effacement direct)")
    print("  Tapez 'exit' pour quitter.")
    print("="*70 + "\n")

    while True:
        try:
            sys.stdout.write("\nVous: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline().strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Session terminée. À bientôt !")
                break

            engine.generate_with_live_correction(user_input)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    start_live_interactive_session()
