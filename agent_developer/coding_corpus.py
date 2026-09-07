"""
Corpus d'Apprentissage Avancé de la Programmation & Ingénierie Logicielle pour Sarah Engine.
Transmet les connaissances avancées du système Antigravity à l'élève Sarah Engine :
- Règle Maîtresse : Tout le code Web HTML/CSS/JS est 100% Single-File (<style> et <script> internes).
- Esthétique Premium : Dark Mode OLED, Glassmorphism, Typographie Apple SF Pro, Micro-interactions.
- Multi-Langages : HTML/CSS/JS, Python (PyTorch/FastAPI), Node.js, Java, Swift/SwiftUI & Apple Shortcuts.
"""

from typing import List, Dict

CODING_SAMPLES: Dict[str, List[str]] = {
    "HTML_CSS_JS_Single_File": [
        # Exemple 1 : Boutique Apple E-Commerce Luxury
        """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sarah Engine - Boutique Ultime</title>
  <style>
    :root {
      --bg: #000000;
      --card-bg: rgba(255, 255, 255, 0.05);
      --card-border: rgba(255, 255, 255, 0.12);
      --accent: #0071e3;
      --text: #f5f5f7;
      --text-sub: #86868b;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }
    nav { position: sticky; top: 0; width: 100%; height: 54px; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px); border-bottom: 1px solid var(--card-border); display: flex; align-items: center; justify-content: space-between; padding: 0 48px; z-index: 100; }
    nav .brand { font-size: 18px; font-weight: 600; letter-spacing: -0.5px; }
    .hero { text-align: center; padding: 100px 24px 60px; }
    .hero h1 { font-size: 64px; font-weight: 700; letter-spacing: -1.5px; background: linear-gradient(180deg, #fff 0%, #a1a1a6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { font-size: 24px; color: var(--text-sub); margin-top: 14px; }
    .product-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 32px; max-width: 1100px; margin: 0 auto; padding: 0 24px 80px; }
    .product-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 28px; padding: 40px; text-align: center; backdrop-filter: blur(30px); transition: transform 0.3s ease, border-color 0.3s; }
    .product-card:hover { transform: translateY(-6px); border-color: rgba(255,255,255,0.25); }
    .product-card h3 { font-size: 26px; font-weight: 600; margin-bottom: 8px; }
    .product-card .price { font-size: 19px; color: var(--text-sub); margin-bottom: 24px; }
    .btn-buy { background: var(--accent); color: #fff; border: none; padding: 12px 28px; border-radius: 980px; font-size: 16px; font-weight: 500; cursor: pointer; transition: background 0.2s; }
    .btn-buy:hover { background: #0077ed; }
    .cart-drawer { position: fixed; top: 0; right: -400px; width: 380px; height: 100vh; background: #111113; border-left: 1px solid var(--card-border); padding: 36px; transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1); z-index: 200; }
    .cart-drawer.active { right: 0; }
  </style>
</head>
<body>
  <nav>
    <div class="brand">Sarah Engine</div>
    <button class="btn-buy" style="padding: 6px 16px; font-size: 14px;" onclick="toggleCart()">Panier (<span id="cart-count">0</span>)</button>
  </nav>
  <main class="hero">
    <h1>Conçu pour l'iPhone 14.</h1>
    <p>Le premier modèle Transformer exécuté localement sans cloud.</p>
  </main>
  <section class="product-grid">
    <article class="product-card">
      <h3>Sarah Engine Pro</h3>
      <div class="price">150 Mo RAM • Inférence 4 100 tok/s</div>
      <button class="btn-buy" onclick="addToCart('Sarah Engine Pro')">Commander</button>
    </article>
    <article class="product-card">
      <h3>Sarah Engine Edge</h3>
      <div class="price">Quantification INT8 • 5.1 Mo Poids</div>
      <button class="btn-buy" onclick="addToCart('Sarah Engine Edge')">Commander</button>
    </article>
  </section>
  <aside class="cart-drawer" id="cart">
    <h2>Votre Panier</h2>
    <div id="cart-items" style="margin: 24px 0; color: var(--text-sub);">Panier vide</div>
    <button class="btn-buy" onclick="toggleCart()" style="background: #333;">Fermer</button>
  </aside>
  <script>
    let cart = [];
    function toggleCart() { document.getElementById('cart').classList.toggle('active'); }
    function addToCart(name) {
      cart.push(name);
      document.getElementById('cart-count').innerText = cart.length;
      document.getElementById('cart-items').innerText = cart.length + ' article(s) : ' + cart.join(', ');
      toggleCart();
    }
  </script>
</body>
</html>""",

        # Exemple 2 : Lecteur de Musique / Audio Synthétiseur Minimaliste
        """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Sarah Engine Sound Matrix</title>
  <style>
    body { background: #080b11; color: #fff; font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
    .player { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 32px; width: 320px; text-align: center; backdrop-filter: blur(20px); }
    .visualizer { height: 80px; display: flex; align-items: flex-end; justify-content: center; gap: 6px; margin: 24px 0; }
    .bar { width: 8px; height: 20px; background: #00f2fe; border-radius: 4px; transition: height 0.1s; }
    button { background: #00f2fe; color: #000; border: none; padding: 12px 28px; border-radius: 20px; font-weight: 700; cursor: pointer; }
  </style>
</head>
<body>
  <div class="player">
    <h2>Sarah Wave Synth</h2>
    <p style="color:#888; font-size:13px;">Génération Sonore Web Audio API</p>
    <div class="visualizer" id="vis">
      <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
    </div>
    <button onclick="playTone()">Générer Fréquence</button>
  </div>
  <script>
    function playTone() {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
      const bars = document.querySelectorAll('.bar');
      bars.forEach(b => b.style.height = Math.floor(Math.random() * 60 + 20) + 'px');
      setTimeout(() => bars.forEach(b => b.style.height = '20px'), 300);
    }
  </script>
</body>
</html>"""
    ],

    "Python_DeepLearning": [
        """# Implémentation du Bloc Transformer Sarah Engine
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
"""
    ],

    "Swift_AppleShortcuts": [
        """// Intégration Native Apple Shortcuts & AppIntents pour Sarah Engine
import AppIntents
import SwiftUI

@available(iOS 16.0, macOS 13.0, *)
struct SarahEngineChatIntent: AppIntent {
    static var title: LocalizedStringResource = "Exécuter Sarah Engine"
    static var description = IntentDescription("Envoie une consigne au réseau neuronal Sarah Engine embarqué.")
    
    @Parameter(title: "Prompt", description: "La question ou commande pour Sarah")
    var prompt: String
    
    func perform() async throws -> some ProvidesDialog {
        let response = await SarahEngineBridge.generate(prompt: prompt)
        return .result(dialog: IntentDialog(stringLiteral: response))
    }
}
"""
    ],

    "NodeJS_Backend": [
        """// Serveur Micro-Service HTTP Node.js Ultra-Rapide pour Sarah Engine
const http = require('http');

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/v1/sarah/infer') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const payload = JSON.parse(body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ model: 'Sarah Engine', status: 'ready', answer: 'Inférence locale exécutée.' }));
    });
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(8080, () => console.log('Serveur Sarah Engine en écoute sur 8080'));
"""
    ]
}

def generate_coding_corpus() -> List[str]:
    """Compile l'ensemble des règles, architectures et connaissances avancées."""
    corpus = []
    for category, snippets in CODING_SAMPLES.items():
        for snippet in snippets:
            corpus.append(f"<knowledge_category>{category}</knowledge_category>\n<source_code>\n{snippet}\n</source_code>")
    return corpus
