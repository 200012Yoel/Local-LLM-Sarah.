"""
Module de nettoyage et de normalisation textuelle pour Sarah Ngin.
"""

import unicodedata
import re
from typing import List, Generator

class TextCleaner:
    def __init__(self):
        # Expressions régulières pour filtrer les caractères parasites
        self.multiple_spaces = re.compile(r"[ \t]+")
        self.multiple_newlines = re.compile(r"\n{3,}")
        self.special_chars_to_clean = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    def normalize(self, text: str) -> str:
        """
        Normalise le texte en Unicode NFC (gestion correcte des accents français, hébreu, chinois).
        Nettoie les espaces multiples et caractères invisibles.
        """
        if not text:
            return ""
        
        # Normalisation Unicode
        text = unicodedata.normalize("NFC", text)
        
        # Suppression des caractères de contrôle
        text = self.special_chars_to_clean.sub("", text)
        
        # Nettoyage des espaces et retours à la ligne
        text = self.multiple_spaces.sub(" ", text)
        text = self.multiple_newlines.sub("\n\n", text)
        
        return text.strip()

    def filter_valid_sentence(self, sentence: str, min_chars: int = 4, max_chars: int = 1500) -> bool:
        """
        Vérifie si une ligne / phrase est utile pour l'apprentissage linguistique.
        """
        cleaned = sentence.strip()
        if len(cleaned) < min_chars or len(cleaned) > max_chars:
            return False
        
        # Doit contenir au moins des lettres ou caractères alphabétiques
        if not re.search(r"[\w\u0590-\u05FF\u4e00-\u9fff]", cleaned, re.UNICODE):
            return False
        
        return True

    def process_corpus(self, raw_lines: List[str]) -> List[str]:
        """
        Nettoie et déduplique une liste de textes.
        """
        seen = set()
        clean_lines = []
        for line in raw_lines:
            norm = self.normalize(line)
            if self.filter_valid_sentence(norm) and norm not in seen:
                seen.add(norm)
                clean_lines.append(norm)
        return clean_lines
