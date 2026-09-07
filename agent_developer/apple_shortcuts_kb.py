"""
Base de Connaissances Exhaustive sur Apple Shortcuts (Raccourcis iOS/macOS) et Xcode Swift.
Couvre l'ensemble des modules d'automatisation Apple officiels et les App Intents.
"""

from typing import List, Dict, Any

APPLE_SHORTCUTS_ACTIONS = {
    "Scripting & Logique": [
        ("Définir une variable (Set Variable)", "Stocke une valeur textuelle, numérique ou objet dans une variable réutilisable."),
        ("Condition Si/Sinon (If/Else)", "Exécute un bloc d'actions si la condition logique est remplie, sinon bascule sur le bloc secondaire."),
        ("Répéter avec chaque élément (Repeat with Each)", "Boucle itérative sur une liste d'éléments (fichiers, textes, contacts)."),
        ("Exécuter JavaScript sur page web (Run JavaScript on Webpage)", "Injecte du code JS directement dans le DOM Safari pour scraper ou modifier une page."),
        ("Exécuter un script via SSH (Run Script over SSH)", "Se connecte à un serveur distant et exécute des commandes shell Linux/macOS.")
    ],
    "Réseau & Requêtes Web / API": [
        ("Obtenir le contenu de l'URL (Get Contents of URL)", "Effectue des requêtes HTTP (GET, POST, PUT, DELETE) avec en-têtes personnalisés et corps JSON."),
        ("Ouvrir une URL (Open URL)", "Ouvre un lien Web ou un schéma d'URL profond (Universal Link) dans le navigateur ou une app dédiée."),
        ("Télécharger le fichier (Download URL File)", "Récupère un flux binaire ou un document distant et l'enregistre dans Fichiers iCloud.")
    ],
    "Système iOS, Presse-papier & Partage": [
        ("Obtenir le presse-papiers (Get Clipboard)", "Lit le texte ou l'image actuellement copiée dans le presse-papier iOS."),
        ("Copier dans le presse-papiers (Set Clipboard)", "Écrit une donnée dans le presse-papiers universel avec expiration optionnelle."),
        ("Afficher une notification (Show Notification)", "Déclenche une bannière système avec titre, corps de texte et son."),
        ("Aperçu rapide (Quick Look)", "Ouvre une fenêtre modale pour prévisualiser instantanément un document, JSON ou image."),
        ("Envoyer un message (Send Message via iMessage/SMS)", "Prépare ou envoie automatiquement un message à un ou plusieurs destinataires.")
    ],
    "Intégration Xcode & Swift (App Intents)": [
        ("Framework AppIntents (iOS 16+)", "Définit des intents natifs en Swift pour exposer des actions de l'application à Siri et Raccourcis."),
        ("AppShortcut (ShortcutsProvider)", "Enregistre automatiquement des raccourcis prêts à l'emploi dès l'installation de l'application sans configuration manuelle."),
        ("IntentParameter", "Expose des paramètres typés (texte, date, énumération) pour permettre à l'utilisateur de les personnaliser dans Raccourcis.")
    ]
}

def generate_apple_shortcuts_corpus() -> List[str]:
    """Génère des données d'entraînement exhaustives sur les raccourcis Apple et Xcode."""
    corpus = []
    
    for category, actions in APPLE_SHORTCUTS_ACTIONS.items():
        for act_name, act_desc in actions:
            corpus.append(f"Apple Shortcuts ({category}) : Action '{act_name}' - Rôle : {act_desc}")
            corpus.append(f"Guide Raccourcis iOS : Pour utiliser '{act_name}', on configure le flux d'automatisation selon : {act_desc}")
            corpus.append(f"Automatisation Apple : '{act_name}' permet d'interagir nativement avec l'écosystème iPhone et macOS.")

    # Exemples de code Swift Xcode pour App Intents
    swift_examples = [
        "import AppIntents\n\nstruct SarahNginIntent: AppIntent {\n    static var title: LocalizedStringResource = 'Demander à Sarah Ngin'\n    @Parameter(title: 'Question') var prompt: String\n    func perform() async throws -> some ProvidesDialog {\n        return .result(dialog: 'Sarah Ngin traite votre demande.')\n    }\n}",
        "import AppIntents\n\nstruct SarahShortcutsProvider: AppShortcutsProvider {\n    static var appShortcuts: [AppShortcut] {\n        AppShortcut(\n            intent: SarahNginIntent(),\n            phrases: ['Demande à Sarah Ngin', 'Ouvre Sarah Ngin']\n        )\n    }\n}",
        "// Apple Shortcuts HTTP API Request:\n// Action: Get Contents of URL -> Method: POST -> Headers: Content-Type: application/json -> Request Body: JSON payload"
    ]
    
    for ex in swift_examples:
        corpus.append(f"Exemple de Code Swift / Xcode pour Apple Shortcuts :\n{ex}")

    return corpus
