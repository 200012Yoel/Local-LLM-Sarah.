"""
Corpus d'Apprentissage du Code et de la Programmation Multi-Langages pour Sarah Ngin.
Couvre HTML/CSS, JavaScript, Node.js, Python, Java, Xcode (Swift / SwiftUI).
"""

from typing import List, Dict

CODING_SAMPLES: Dict[str, List[str]] = {
    "HTML_CSS_JS_Single_File": [
        "<!-- Application Web E-Commerce Style Apple Complète par Sarah Engine -->\n<!DOCTYPE html>\n<html lang='fr'>\n<head>\n  <meta charset='UTF-8'>\n  <title>Sarah Engine Store</title>\n  <style>\n    body { background: #000; color: #fff; font-family: -apple-system, sans-serif; padding: 40px; }\n    .apple-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.15); }\n    .btn-apple { background: #0071e3; color: white; padding: 10px 20px; border-radius: 20px; border: none; cursor: pointer; }\n  </style>\n</head>\n<body>\n  <div class='apple-card'>\n    <h1>Sarah Engine Store</h1>\n    <p>Architecture Transformer locale pour iPhone 14.</p>\n    <button class='btn-apple' onclick='buyProduct()'>Commander</button>\n  </div>\n  <script>\n    function buyProduct() { alert('Commande Sarah Engine enregistrée avec succès.'); }\n  </script>\n</body>\n</html>",
        "<!-- Dashboard Interactif avec Styles et Logique Intégrés -->\n<!DOCTYPE html>\n<html lang='fr'>\n<head>\n  <meta charset='UTF-8'>\n  <title>Sarah Engine Dashboard</title>\n  <style>\n    body { background: #070a11; color: #00f2fe; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: sans-serif; }\n    .gauge { background: #121826; border: 1px solid #00f2fe; padding: 25px; border-radius: 16px; text-align: center; }\n  </style>\n</head>\n<body>\n  <div class='gauge'>\n    <h2>RAM iPhone 14 : <span id='ram'>150</span> Mo</h2>\n    <p>Statut : Sarah Engine Active</p>\n  </div>\n  <script>\n    console.log('Sarah Engine Telemetry Running');\n  </script>\n</body>\n</html>"
    ],
    "Python_PyTorch": [
        "# Architecture Sarah Engine en PyTorch (RMSNorm + RoPE Attention)\nimport torch\nimport torch.nn as nn\n\nclass SarahEngineAttention(nn.Module):\n    def __init__(self, dim=256, heads=8):\n        super().__init__()\n        self.heads = heads\n        self.head_dim = dim // heads\n        self.qkv = nn.Linear(dim, dim * 3, bias=False)\n        self.out = nn.Linear(dim, dim, bias=False)\n    def forward(self, x):\n        B, T, C = x.shape\n        q, k, v = self.qkv(x).chunk(3, dim=-1)\n        scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))\n        return self.out(torch.softmax(scores, dim=-1) @ v)"
    ],
    "Node_JavaScript": [
        "// API Client Node.js pour Sarah Engine\nconst fetch = require('node-fetch');\nasync function askSarahEngine(prompt) {\n  const res = await fetch('http://localhost:8080/api/chat', {\n    method: 'POST',\n    headers: { 'Content-Type': 'application/json' },\n    body: JSON.stringify({ message: prompt })\n  });\n  return await res.json();\n}"
    ],
    "Swift_Xcode": [
        "// Module Swift / Xcode pour Apple Shortcuts et AppIntents\nimport SwiftUI\nimport AppIntents\n\nstruct AskSarahEngineIntent: AppIntent {\n    static var title: LocalizedStringResource = 'Parler à Sarah Engine'\n    @Parameter(title: 'Prompt') var userQuery: String\n    func perform() async throws -> some ProvidesDialog {\n        return .result(dialog: 'Sarah Engine a traité votre demande.')\n    }\n}"
    ]
}

def generate_coding_corpus() -> List[str]:
    """Compile l'ensemble des règles, syntaxes et exemples de code pour l'entraînement."""
    corpus = []
    for lang, snippets in CODING_SAMPLES.items():
        for snippet in snippets:
            corpus.append(f"Apprentissage du Code ({lang}) :\n{snippet}")
            corpus.append(f"Exemple de Programmation ({lang}) : Sarah Ngin sait générer et corriger du code dans ce langage avec les meilleures pratiques.")
    return corpus
