"""
Corpus de Connaissances Polyglotte & Patterns de Programmation Zéro-Erreur.
Transmis par Antigravity (Superviseur) pour l'apprentissage intensif de Sarah Engine.
"""

from typing import List

def generate_coding_corpus() -> List[str]:
    corpus = []

    # 1. Principes Fondamentaux de Programmation & 0-Erreur
    principles = [
        "Règle de fiabilité : Tout code généré doit être syntaxiquement valide, typé, et exempt de bugs.",
        "Architecture Single-File Web : Les composants HTML, le style CSS et la logique JavaScript doivent être unifiés dans un seul fichier autonome.",
        "Gestion de la mémoire : Éviter les fuites de mémoire, libérer les écouteurs d'événements et utiliser des structures légères.",
        "Complexité algorithmique : Privilégier les algorithmes en O(n) ou O(n log n) et éviter les boucles imbriquées inutiles.",
        "Typage et Clarté : Déclarer explicitement les types et documenter les entrées et sorties des fonctions."
    ]
    corpus.extend(principles)

    # 2. Patterns Python (Calcul Scientifique, IA, Algorithmes)
    python_patterns = [
        "Python : Calcul de la perte d'entropie croisée : loss = -torch.sum(targets * torch.log(predictions + 1e-12)) / batch_size",
        "Python : Définition d'un module PyTorch Transformer : class TransformerBlock(nn.Module): def __init__(self, d_model, n_heads): super().__init__()",
        "Python : Fonction de recherche binaire optimisée : def binary_search(arr: list, target: int) -> int: left, right = 0, len(arr) - 1",
        "Python : Décorateur de mesure de latence : def timeit(func): def wrapper(*args, **kwargs): start = time.perf_counter(); return func(*args, **kwargs)",
        "Python : Générateur de tokens avec streaming : def stream_tokens(model, tokenizer, prompt): for token in model.generate(prompt): yield token",
        "Python : Optimiseur AdamW avec découplage du Weight Decay pour stabiliser l'apprentissage des réseaux de neurones.",
        "Python : Mécanisme d'attention multi-têtes : Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V",
        "Python : Quantification INT8 : torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)"
    ]
    corpus.extend(python_patterns)

    # 3. Patterns Front-End & Architecture Single-File (HTML5, CSS3, JavaScript ES6+)
    web_patterns = [
        "HTML/CSS/JS : Structure Single-File propre avec <!DOCTYPE html><html><head><style>...</style></head><body>...<script>...</script></body></html>",
        "CSS Moderne : Utilisation des variables CSS :root { --bg: #0d0e12; --accent: #3b82f6; --text: #f3f4f6; } pour le mode sombre.",
        "CSS Glassmorphism : backdrop-filter: blur(20px); background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);",
        "JavaScript : Requête asynchrone sécurisée : const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }) });",
        "JavaScript Web Audio API : const ctx = new AudioContext(); const osc = ctx.createOscillator(); osc.frequency.setValueAtTime(440, ctx.currentTime);",
        "JavaScript HTML5 Canvas : const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, width, height); ctx.arc(x, y, radius, 0, Math.PI * 2);",
        "JavaScript Gestion d'événements clavier : window.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) sendMessage(); });"
    ]
    corpus.extend(web_patterns)

    # 4. Patterns Swift & SwiftUI pour iOS (iPhone 14 / CoreML)
    swift_patterns = [
        "SwiftUI : Structure de vue principale : struct ContentView: View { @State private var message = ''; var body: some View { VStack { Text(message) } } }",
        "SwiftUI : Gestion de la mémoire et des états : @StateObject private var modelEngine = SarahEngineViewModel()",
        "Swift : Intégration de modèle CoreML ou TorchScript Mobile : let module = TorchModule(fileAtPath: modelPath)",
        "Swift : Exécution en arrière-plan asynchrone : Task.detached(priority: .userInitiated) { let output = await engine.generate(prompt: prompt) }",
        "Swift : Performance sur iPhone 14 : Consommation mémoire optimisée sous 150 Mo avec quantification des tenseurs."
    ]
    corpus.extend(swift_patterns)

    # 5. Patterns Java & Orienté Objet Robuste
    java_patterns = [
        "Java : Modèle de données immuable : public record UserMemory(String id, String text, double importance) {}",
        "Java : Traitement fonctionnel avec Stream API : list.stream().filter(m -> m.importance() > 1.5).map(UserMemory::text).toList();",
        "Java : Singleton Thread-Safe pour moteur d'inférence : public class ModelEngine { private static volatile ModelEngine instance; }",
        "Java : Gestion propre des exceptions : try (var reader = Files.newBufferedReader(path)) { return reader.lines().toList(); }"
    ]
    corpus.extend(java_patterns)

    # 6. Paires Question/Réponse Code pour l'entraînement Superviseur <-> Élève
    qa_pairs = [
        "Question: Comment optimiser un réseau de neurones pour mobile ?\nRéponse: Appliquer la quantification dynamique INT8, réduire la dimension d'embedding et fusionner les opérations linéaires.",
        "Question: Quelle est la règle d'or pour un front-end Single-File ?\nRéponse: Intégrer l'ensemble du HTML, des styles CSS modernes et des scripts JS dans un seul fichier sans dépendance externe superflue.",
        "Question: Comment fonctionne l'attention multi-têtes ?\nRéponse: Elle projette les vecteurs Query, Key et Value dans plusieurs sous-espaces pour capturer simultanément différentes relations sémantiques.",
        "Question: Comment garantir 0 erreur dans la génération de code ?\nRéponse: Respecter scrupuleusement la syntaxe du langage, valider l'AST, typer les variables et tester chaque fonction de manière unitaire.",
        "Question: Pourquoi Sarah Ngin est-elle optimisée pour iPhone 14 ?\nRéponse: Avec 3.71M de paramètres et une empreinte mémoire inférieure à 150 Mo, elle s'exécute localement à 3400+ tokens/s sur 4 Go de RAM."
    ]
    
    # Répéter les paires fondamentales pour ancrer les poids neuronaux
    for qa in qa_pairs:
        for _ in range(30):
            corpus.append(qa)

    return corpus
