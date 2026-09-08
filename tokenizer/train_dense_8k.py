"""
Entraînement du Tokenizer BPE Ultra-Dense (8 192 tokens) pour Sarah Engine.
Couvre Français littéraire, Hébreu bilingue, Anglais, Argot/SMS et Code informatique.
"""

import sys
from pathlib import Path

# Fix stdout encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tokenizer.bpe_tokenizer import SarahTokenizer
from agent_developer.argot_sms_dict import ARGOT_SMS_DICTIONARY

def build_multilingual_corpus() -> list[str]:
    corpus = []
    
    # 1. Livres classiques en français
    for txt_file in Path("data/raw").glob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="ignore")
            lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 5]
            corpus.extend(lines[:1500])
        except Exception as e:
            print(f"Avertissement lecture {txt_file}: {e}")

    # 2. Dictionnaire Argot / SMS
    for slang, (trad, ex, defn) in ARGOT_SMS_DICTIONARY.items():
        corpus.append(f"Argot: {slang} signifie {trad}. Exemple: {ex}. Définition: {defn}")

    # 3. Dictionnaire Hébreu / Français
    hebrew_pairs = [
        ("חָכְמָה", "La sagesse", "Hokhmah"),
        ("שָׁלוֹם", "La paix", "Shalom"),
        ("אֱלֹהִים", "Dieu", "Elohim"),
        ("בְּרֵאשִׁית", "Au commencement", "Bereshit"),
        ("תּוֹרָה", "La Loi, l'enseignement", "Torah"),
        ("אַהֲבָה", "L'amour", "Ahava"),
        ("אֱמֶת", "La vérité", "Emet"),
        ("חַיִּים", "La vie", "Hayim"),
        ("אוֹר", "La lumière", "Or"),
        ("מַלְאָךְ", "Un messager, un ange", "Malakh")
    ]
    for heb, fr, phon in hebrew_pairs:
        corpus.append(f"Hébreu: {heb} ({phon}) se traduit par {fr} en français.")

    # 4. Spécifications & Code Informatique (Python, JS, XML, PHP, Swift, Java, HTML)
    code_snippets = [
        "def compute_loss(pred, target): return torch.nn.functional.cross_entropy(pred, target)",
        "async function fetchSarahEngine(prompt) { const res = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({ message: prompt }) }); }",
        "<?xml version='1.0' encoding='UTF-8'?><sarah_engine><model name='Sarah Ngin' version='2.0'/></sarah_engine>",
        "<?php function callSarahEngine($prompt) { return file_get_contents('http://localhost:8080/api/chat'); } ?>",
        "import SwiftUI\nstruct SarahView: View { @State var text: String = '' var body: some View { Text(text) } }",
        "public class SarahEngine { public static void main(String[] args) { System.out.println('Sarah MoE BitNet Ready'); } }",
        "<div class='chat-box' id='chatContainer'><input type='text' id='userInput' placeholder='Message...'/></div>"
    ]
    corpus.extend(code_snippets * 50)

    return corpus

def train_8k_tokenizer():
    print("=== ENTRAÎNEMENT DU TOKENIZER BPE DENSE (8 192 TOKENS) ===")
    corpus = build_multilingual_corpus()
    print(f"Taille du corpus d'entraînement : {len(corpus)} segments.")

    tokenizer = SarahTokenizer(vocab_size=8192)
    tokenizer.train(corpus)

    out_path = "checkpoints/sarah_tokenizer_8k.json"
    tokenizer.save(out_path)
    print(f"✓ Tokenizer 8k sauvegardé avec succès dans : {out_path} (Vocabulaire effectif : {tokenizer.vocab_size})")

if __name__ == "__main__":
    train_8k_tokenizer()
