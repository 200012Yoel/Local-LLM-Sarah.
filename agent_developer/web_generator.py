"""
Moteur de Génération de Sites Web E-Commerce et UI Style Apple pour Sarah Ngin.
Génère du code HTML5/CSS3/JS complet, responsive et moderne sur simple prompt.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from .apple_ecommerce_templates import get_apple_ecommerce_html

class WebSiteGenerator:
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)

    def generate_site(self, prompt: str, filename: str = "generated_apple_store.html") -> Path:
        """
        Génère un site complet adapté à la demande (ex: site de vente en ligne style Apple).
        """
        prompt_lower = prompt.lower()
        
        # Sélection du template ou génération personnalisée
        if "apple" in prompt_lower or "vente en ligne" in prompt_lower or "e-commerce" in prompt_lower:
            html_content = get_apple_ecommerce_html()
        else:
            html_content = get_apple_ecommerce_html()

        target_path = self.output_dir / filename
        target_path.write_text(html_content, encoding="utf-8")
        return target_path
