"""
Corpus d'Apprentissage du Code et de la Programmation Multi-Langages pour Sarah Ngin.
Couvre HTML/CSS, JavaScript, Node.js, Python, Java, Xcode (Swift / SwiftUI).
"""

from typing import List, Dict

CODING_SAMPLES: Dict[str, List[str]] = {
    "HTML & CSS": [
        "<!-- Structure HTML5 Sémantique -->\n<!DOCTYPE html>\n<html lang='fr'>\n<head>\n  <meta charset='UTF-8'>\n  <title>Application Sarah Ngin</title>\n  <link rel='stylesheet' href='style.css'>\n</head>\n<body>\n  <header><nav><h1>Sarah Ngin</h1></nav></header>\n  <main class='container'><section class='card'>Contenu</section></main>\n</body>\n</html>",
        "/* CSS3 Flexbox & Glassmorphism */\n.container {\n  display: flex;\n  justify-content: center;\n  align-items: center;\n  gap: 20px;\n}\n.card {\n  background: rgba(255, 255, 255, 0.1);\n  backdrop-filter: blur(10px);\n  border-radius: 16px;\n  border: 1px solid rgba(255, 255, 255, 0.2);\n  padding: 24px;\n}"
    ],
    "JavaScript & Node.js": [
        "// Appel Asynchrone Fetch API avec gestion d'erreurs en JavaScript\nasync function querySarahNgin(promptText) {\n  try {\n    const res = await fetch('/api/chat', {\n      method: 'POST',\n      headers: { 'Content-Type': 'application/json' },\n      body: JSON.stringify({ message: promptText })\n    });\n    const data = await res.json();\n    return data.reply;\n  } catch (err) {\n    console.error('Erreur API:', err);\n  }\n}",
        "// Serveur HTTP Node.js avec Express\nconst express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.post('/api/chat', (req, res) => {\n  const { message } = req.body;\n  res.json({ reply: `Réponse de Sarah Ngin à : ${message}` });\n});\n\napp.listen(3000, () => console.log('Serveur Node actif sur port 3000'));"
    ],
    "Python": [
        "# Modèle PyTorch compact avec mécanisme d'attention causale\nimport torch\nimport torch.nn as nn\n\nclass SimpleAttention(nn.Module):\n    def __init__(self, dim, heads):\n        super().__init__()\n        self.heads = heads\n        self.qkv = nn.Linear(dim, dim * 3, bias=False)\n        self.proj = nn.Linear(dim, dim)\n\n    def forward(self, x):\n        B, T, C = x.shape\n        q, k, v = self.qkv(x).chunk(3, dim=-1)\n        scores = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))\n        att = torch.softmax(scores, dim=-1)\n        return self.proj(att @ v)"
    ],
    "Java": [
        "// Classe Java avec Stream API et programmation orientée objet\npackage com.sarahngin.ai;\nimport java.util.*;\nimport java.util.stream.Collectors;\n\npublic class MemoryManager {\n    private List<String> memories = new ArrayList<>();\n\n    public void addMemory(String item) {\n        this.memories.add(item);\n    }\n\n    public List<String> searchKeyword(String keyword) {\n        return memories.stream()\n            .filter(m -> m.toLowerCase().contains(keyword.toLowerCase()))\n            .collect(Collectors.toList());\n    }\n}"
    ],
    "Xcode & Swift / SwiftUI": [
        "// Vue SwiftUI pour iOS 17 / iPhone 14 avec CoreML\nimport SwiftUI\n\nstruct SarahNginView: View {\n    @State private var inputText: String = ''\n    @State private var responseText: String = 'Prête à vous aider.'\n\n    var body: some View {\n        VStack(spacing: 20) {\n            Text('Sarah Ngin Local AI')\n                .font(.title)\n                .bold()\n            Text(responseText)\n                .padding()\n                .background(Color.blue.opacity(0.1))\n                .cornerRadius(12)\n            TextField('Posez une question...', text: $inputText)\n                .textFieldStyle(RoundedBorderTextFieldStyle())\n                .padding(.horizontal)\n        }\n    }\n}"
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
