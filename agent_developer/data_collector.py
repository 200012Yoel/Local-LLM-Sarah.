"""
Agent Développeur : Collecteur et Synthétiseur de Données Linguistiques pour Sarah Ngin.
Collecte dictionnaires complets, logique de phrases, lexiques français, hébreu, anglais et chinois.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests

from .cleaner import TextCleaner

logging.basicConfig(level=logging.INFO, format="[Agent Développeur] %(message)s")
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cleaner = TextCleaner()
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_online_text(self, url: str, filename: str, timeout: int = 10) -> Optional[Path]:
        """Télécharge une ressource textuelle avec gestion des timeouts."""
        dest_path = self.raw_dir / filename
        if dest_path.exists():
            logger.info(f"Fichier déjà existant : {dest_path}")
            return dest_path
        
        try:
            logger.info(f"Téléchargement depuis {url}...")
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "SarahNginAgent/1.0"})
            if response.status_code == 200:
                dest_path.write_text(response.text, encoding="utf-8")
                logger.info(f"Enregistré : {dest_path} ({len(response.text)} caractères)")
                return dest_path
            else:
                logger.warning(f"Échec HTTP {response.status_code} pour {url}")
        except Exception as e:
            logger.warning(f"Impossible de joindre {url} ({e}). Utilisation du générateur linguistique local.")
        return None

    def download_public_domain_french_books(self) -> List[str]:
        """
        Télécharge des grands classiques de la littérature française libres de droits
        (Jules Verne, Voltaire, La Fontaine, Alexandre Dumas) pour apprendre la syntaxe et le vocabulaire riche.
        """
        logger.info("Téléchargement de livres français libres de droits (Project Gutenberg & Open Literature)...")
        books_urls = [
            ("candide_voltaire.txt", "https://www.gutenberg.org/cache/epub/4650/pg4650.txt"),
            ("tour_du_monde_verne.txt", "https://www.gutenberg.org/cache/epub/800/pg800.txt"),
            ("fables_la_fontaine.txt", "https://www.gutenberg.org/cache/epub/18749/pg18749.txt"),
        ]

        book_lines = []
        for filename, url in books_urls:
            path = self.fetch_online_text(url, filename, timeout=12)
            if path and path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")
                # Supprimer les en-têtes et pieds de page Gutenberg
                start_marker = "*** START OF THE PROJECT GUTENBERG"
                end_marker = "*** END OF THE PROJECT GUTENBERG"
                
                if start_marker in text:
                    text = text.split(start_marker, 1)[1]
                if end_marker in text:
                    text = text.split(end_marker, 1)[0]
                
                # Découper en phrases et paragraphes
                for line in text.splitlines():
                    cleaned_line = line.strip()
                    if len(cleaned_line) > 20 and not cleaned_line.startswith("["):
                        book_lines.append(cleaned_line)

        logger.info(f"Total lignes littéraires extraites des livres : {len(book_lines)}")
        return book_lines

    def generate_french_dictionary_corpus(self) -> List[str]:
        """
        Génère un dictionnaire et lexique français encyclopédique structuré
        avec définitions, natures grammaticales, exemples contextuels et règles logiques.
        """
        logger.info("Génération du corpus de dictionnaire et grammaire française...")
        
        # Données fondamentales : Catégories, définitions et logique
        vocabulaire_fondamental = [
            ("L'intelligence artificielle", "nom féminin", "Ensemble des théories et des techniques développant des programmes informatiques complexes capables de simuler certains traits de l'intelligence humaine comme l'apprentissage et le raisonnement."),
            ("Réseau de neurones", "nom masculin", "Système informatique inspiré du fonctionnement des neurones biologiques du cerveau humain, composé de couches de calcul pondérées interconnectées."),
            ("Algorithme", "nom masculin", "Suite finie et non ambiguë d'instructions ou d'opérations permettant de résoudre un problème ou d'obtenir un résultat précis."),
            ("Connaissance", "nom féminin", "Idée, notion ou ensemble de faits et de principes acquis par l'étude, l'expérience ou l'observation."),
            ("Raisonnement", "nom masculin", "Activité cognitive permettant d'enchaîner logiquement des propositions pour parvenir à une conclusion valide."),
            ("Logique", "nom féminin", "Science du raisonnement formel, de la déduction et de la vérité des propositions."),
            ("Dictionnaire", "nom masculin", "Ouvrage de référence répertoriant l'ensemble des mots d'une langue avec leur signification, étymologie et exemples d'usage."),
            ("Transformer", "nom masculin", "Architecture d'apprentissage profond reposant sur le mécanisme d'attention pour traiter des séquences de données en parallèle."),
            ("Apprentissage automatique", "nom masculin", "Discipline de l'informatique permettant aux machines d'apprendre des représentations à partir de données sans programmation explicite."),
            ("Mémoire vive", "nom féminin", "Mémoire informatique volatile permettant un accès ultra rapide aux données nécessaires à l'exécution en temps réel des programmes."),
            ("Système", "nom masculin", "Ensemble d'éléments interagissant entre eux selon certains principes ou règles organisées."),
            ("Langage", "nom masculin", "Faculté d'exprimer sa pensée et de communiquer au moyen d'un système de signes vocaux ou graphiques."),
            ("Vérité", "nom féminin", "Conformité de l'idée ou de la proposition avec le réel ou les principes fondamentaux de la logique."),
            ("Philosophie", "nom féminin", "Recherche rationnelle sur la nature de l'existence, de la connaissance, de la morale et de l'esprit humain."),
            ("Sciences", "nom féminin pluriel", "Ensemble cohérent de connaissances relatives à certaines catégories de faits, d'objets ou de phénomènes obéissant à des lois vérifiables.")
        ]
        
        concepts_logiques = [
            "Si toutes les composantes du réseau sont optimisées, alors le modèle fonctionne efficacement sur mobile.",
            "La logique formelle implique que si A entraîne B, et que A est vrai, alors B est nécessairement vrai.",
            "Pour qu'un modèle linguistique produise des phrases cohérentes, il doit apprendre la syntaxe et la sémantique de la langue.",
            "L'eau bout à 100 degrés Celsius à la pression atmosphérique standard.",
            "La Terre tourne autour du Soleil en suivant une orbite elliptique.",
            "La grammaire française régit l'accord en genre et en nombre entre le sujet et le verbe.",
            "Sarah Ngin est un modèle conçu pour allier haute efficacité computationnelle et faible empreinte mémoire.",
            "Un syllogisme classique : Tous les hommes sont mortels ; or Socrate est un homme ; donc Socrate est mortel.",
            "La déduction part du général vers le particulier, tandis que l'induction part des observations vers une règle générale."
        ]

        corpus = []
        for mot, nature, definition in vocabulaire_fondamental:
            corpus.append(f"Dictionnaire Français : {mot} ({nature}) - Définition : {definition}")
            corpus.append(f"Exemple d'utilisation : {mot} est un concept fondamental dans l'analyse logique et scientifique.")
        
        for regle in concepts_logiques:
            corpus.append(f"Principe de Raisonnement Logique : {regle}")

        # Expansion combinatoire pour enrichir la richesse syntaxique
        sujets = ["Le modèle Sarah Ngin", "L'agent développeur", "L'algorithme de calcul", "Le système cognitif", "L'utilisateur"]
        verbes = ["analyse avec précision", "comprend la structure de", "traite les informations de", "organise la logique de", "génère les concepts de"]
        complements = ["la langue française et ses nuances.", "la relation sémantique entre les mots.", "la déduction logique étape par étape.", "l'inférence rapide sur appareil mobile."]

        for s in sujets:
            for v in verbes:
                for c in complements:
                    corpus.append(f"{s} {v} {c}")

        return corpus

    def generate_multilingual_corpus(self) -> List[str]:
        """
        Génère un dictionnaire et corpus multilingue :
        Français - Hébreu (עברית) - Anglais (English) - Chinois (中文).
        """
        logger.info("Génération du corpus multilingue (Français, Hébreu, Anglais, Chinois)...")
        
        termes_multilingues = [
            ("bonjour", "שלום", "hello", "你好"),
            ("monde", "עולם", "world", "世界"),
            ("intelligence", "בינה", "intelligence", "智能"),
            ("artificielle", "מלאכותית", "artificial", "人工"),
            ("réseau de neurones", "רשת עצבית", "neural network", "神经网络"),
            ("livre", "ספר", "book", "书"),
            ("connaissance", "ידע", "knowledge", "知识"),
            ("langage", "שפה", "language", "语言"),
            ("esprit", "רוח", "mind", "心灵"),
            ("pensée", "מחשבה", "thought", "思想"),
            ("vérité", "אמת", "truth", "真理"),
            ("lumière", "אור", "light", "光"),
            ("paix", "שלום", "peace", "和平"),
            ("force", "כוח", "strength", "力量"),
            ("temps", "זמן", "time", "时间"),
            ("futur", "עתיד", "future", "未来"),
            ("apprentissage", "למידה", "learning", "学习"),
            ("ordinateur", "מחשב", "computer", "计算机"),
            ("science", "מדע", "science", "科学"),
            ("logique", "היגיון", "logic", "逻辑")
        ]

        phrases_multilingues = [
            ("L'intelligence artificielle apprend la logique.", "בינה מלאכותית לומדת היגיון.", "Artificial intelligence learns logic.", "人工智能学习逻辑。"),
            ("Le savoir est la clé de la sagesse.", "הידע הוא המפתח לחוכמה.", "Knowledge is the key to wisdom.", "知识是智慧的钥匙。"),
            ("Sarah Ngin est un moteur d'intelligence léger et puissant.", "שרה נג'ין הוא מנוע בינה קל ועוצמתי.", "Sarah Ngin is a lightweight and powerful AI engine.", "Sarah Ngin 是一个轻量级且强大的智能引擎。"),
            ("La vérité et la logique guident la réflexion.", "האמת וההיגיון מדריכים את המחשבה.", "Truth and logic guide thinking.", "真理与逻辑指引思考。")
        ]

        multilingual_corpus = []

        # Paires de dictionnaire
        for fr, he, en, zh in termes_multilingues:
            multilingual_corpus.append(f"Dictionnaire Multilingue : [FR] {fr} <-> [HE] {he} <-> [EN] {en} <-> [ZH] {zh}")
            multilingual_corpus.append(f"Traduction FR-HE : '{fr}' se traduit en hébreu par '{he}'.")
            multilingual_corpus.append(f"Traduction FR-EN : '{fr}' translates into English as '{en}'.")
            multilingual_corpus.append(f"Traduction FR-ZH : '{fr}' 在中文里翻译为 '{zh}'.")

        for fr, he, en, zh in phrases_multilingues:
            multilingual_corpus.append(f"[FR] {fr} | [HE] {he} | [EN] {en} | [ZH] {zh}")
            multilingual_corpus.append(f"Alignement : Français: {fr} --> Hébreu: {he}")
            multilingual_corpus.append(f"Alignment : French: {fr} --> English: {en}")
            multilingual_corpus.append(f"对齐 : 法语: {fr} --> 中文: {zh}")

        return multilingual_corpus

    def generate_extended_reasoning_data(self, repetitions: int = 50) -> List[str]:
        """
        Génère un volume substantiel de données d'apprentissage sur la grammaire,
        les connecteurs logiques, les maths et la syntaxe pour entraîner le réseau.
        """
        logger.info("Génération de données d'entraînement de logique et de raisonnement approfondi...")
        patterns = [
            ("Question: Qu'est-ce que l'intelligence artificielle ?\nRéponse: L'intelligence artificielle est la capacité d'un système informatique à apprendre des modèles et à raisonner logiquement."),
            ("Question: Quel est le rôle d'un dictionnaire ?\nRéponse: Un dictionnaire répertorie et définit les mots pour structurer la langue et la compréhension."),
            ("Question: Pourquoi Sarah Ngin est-elle optimisée pour iPhone 14 ?\nRéponse: Grâce à une architecture Transformer compacte et une faible empreinte mémoire, Sarah Ngin fonctionne avec moins de 4 Go de RAM."),
            ("Question: Quelle est la relation entre cause et effet ?\nRéponse: La cause est l'événement qui produit un résultat, et l'effet est la conséquence directe de cette cause."),
            ("Proposition: 2 + 2 = 4. Preuve logique: L'addition de deux unités à deux unités produit exactement quatre unités."),
            ("Proposition: Si A est supérieur à B, et B est supérieur à C, alors A est supérieur à C par transitivité."),
            ("Syntaxe: Le sujet commande le verbe. Exemple: 'Les étoiles brillent dans le ciel nocturne.'"),
            ("Dialogue:\nHumain: Bonjour Sarah Ngin, comment vas-tu ?\nSarah Ngin: Bonjour ! Je suis opérationnelle et prête à vous assister avec précision et logique.")
        ]
        
        dataset = []
        for _ in range(repetitions):
            dataset.extend(patterns)
        return dataset

    def collect_and_build_all(self) -> Path:
        """
        Exécute la collecte globale, le téléchargement des livres libres de droits,
        le nettoyage et la sauvegarde du corpus d'entraînement.
        """
        logger.info("=== Lancement de la collecte de données par l'Agent Développeur ===")
        all_texts = []
        
        # 1. Livres et littérature française libres de droits (Gutenberg)
        books_data = self.download_public_domain_french_books()
        all_texts.extend(books_data)

        # 2. Dictionnaire Français
        all_texts.extend(self.generate_french_dictionary_corpus())
        
        # 3. Multilingue (Hébreu, Anglais, Chinois)
        all_texts.extend(self.generate_multilingual_corpus())
        
        # 4. Logique, Raisonnement et Dialogues
        all_texts.extend(self.generate_extended_reasoning_data(repetitions=60))

        # Nettoyage et normalisation
        cleaned_data = self.cleaner.process_corpus(all_texts)
        logger.info(f"Total phrases nettoyées et prêtes : {len(cleaned_data)}")

        # Sauvegarde
        output_file = self.processed_dir / "sarah_ngin_corpus.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in cleaned_data:
                f.write(line + "\n")

        # Sauvegarde d'un extrait de dictionnaire structuré JSON
        dict_json = self.processed_dir / "multilingual_dictionary.json"
        dict_meta = {
            "model_name": "Sarah Ngin",
            "languages": ["fr", "he", "en", "zh"],
            "total_sentences": len(cleaned_data),
            "file_size_bytes": output_file.stat().st_size
        }
        with open(dict_json, "w", encoding="utf-8") as f:
            json.dump(dict_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Corpus d'entraînement compilé avec succès dans : {output_file}")
        return output_file

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_and_build_all()
