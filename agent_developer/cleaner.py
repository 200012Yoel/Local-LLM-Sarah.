"""
Nettoyeur et Normalisateur de Texte pour Sarah Ngin.
"""

import re
import unicodedata

class TextCleaner:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Nettoie le texte en normalisant les espaces et les caractères spéciaux."""
        if not text:
            return ""
        # Normalisation Unicode
        text = unicodedata.normalize("NFC", text)
        # Nettoyage des espaces multiples et retours chariots
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    def clean_sentence(self, sentence: str) -> str:
        """Nettoie une phrase unique."""
        cleaned = self.clean_text(sentence)
        # Supprime les sauts de ligne internes
        return " ".join(cleaned.split())

    def process_corpus(self, texts: list) -> list:
        """Nettoie et déduplique un ensemble de lignes textuelles."""
        seen = set()
        cleaned_lines = []
        for text in texts:
            if not text:
                continue
            for line in text.splitlines():
                c = self.clean_sentence(line)
                if len(c) > 5 and c not in seen:
                    seen.add(c)
                    cleaned_lines.append(c)
        return cleaned_lines

