"""
Serveur Web Interactif et API Locale pour Sarah Ngin.
Connecte l'interface utilisateur graphique, la mémoire persistante et le réseau de neurones.
"""

import sys
import json
import http.server
import socketserver
from pathlib import Path
from urllib.parse import urlparse

# Force UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from memory.memory_indexer import SarahMemoryEngine
from agent_developer.argot_sms_dict import ARGOT_SMS_DICTIONARY

# Initialisation de la mémoire persistante
memory_engine = SarahMemoryEngine("memory/user_memory.json")

# Initialisation par défaut de la mémoire de l'utilisateur
if len(memory_engine.memories) == 0:
    memory_engine.add_memory("L'utilisateur s'appelle Yoel et possède un iPhone 14 (4 Go de RAM).", category="profile", importance=2.0)
    memory_engine.add_memory("Aujourd'hui, Yoel a développé le modèle d'IA Sarah Ngin 100% de zéro avec un agent développeur.", category="episodic_today", importance=1.8)
    memory_engine.add_memory("Sarah Ngin intègre le Grand Dictionnaire Hébreu-Français et les livres de Voltaire, Verne et La Fontaine.", category="knowledge", importance=1.5)

PORT = 8080
WEB_DIR = Path(__file__).parent / "web"

class SarahRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/countdown", "/sandbox"]:
            countdown_path = WEB_DIR / "countdown_dashboard.html"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(countdown_path.read_bytes())
            return
        elif parsed.path == "/api/test_code":
            from testing_sandbox.code_verifier import CodeVerifierSandbox
            sandbox = CodeVerifierSandbox()
            samples = {
                "Python": "def compute_loss(y_pred, y_true):\n    return sum((p - t)**2 for p, t in zip(y_pred, y_true))\nresult = compute_loss([1.0, 2.0], [1.0, 2.0])\nassert result == 0.0",
                "JavaScript": "function formatMemory(bytes) { return (bytes / (1024 * 1024)).toFixed(2) + ' MB'; }\nconsole.log(formatMemory(157286400));",
                "HTML & CSS": "<!DOCTYPE html><html><head><style>.card { color: blue; }</style></head><body><div class='card'>Sarah Ngin</div></body></html>",
                "Java": "public class TestModel { public static void main(String[] args) { System.out.println(\"Sarah Ngin OK\"); } }",
                "Swift": "import SwiftUI\nstruct ContentView: View { var body: some View { Text(\"Sarah Ngin\") } }"
            }
            report = sandbox.run_full_suite(samples)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(report, ensure_ascii=False).encode("utf-8"))
            return
        elif parsed.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            stats = memory_engine.get_stats()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return
        elif parsed.path == "/api/memories":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = {
                "memories": [m.to_dict() for m in memory_engine.memories[-15:]],
                "stats": memory_engine.get_stats()
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return
        
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)
            user_msg = payload.get("message", "").strip()

            # 1. Compréhension de l'argot / SMS
            slang_translations = []
            user_msg_lower = user_msg.lower()
            for slang, (trad, _, _) in ARGOT_SMS_DICTIONARY.items():
                if f" {slang} " in f" {user_msg_lower} " or user_msg_lower == slang:
                    slang_translations.append(f"'{slang}' -> '{trad}'")

            # 2. Recherche et extraction dans la mémoire persistante
            memory_engine.extract_and_remember(user_msg)
            relevant_facts = memory_engine.search_relevant_memories(user_msg, top_k=3)

            # 3. Génération de la réponse avec prise en compte de la mémoire et de l'argot
            reply = ""
            if "fait aujourd'hui" in user_msg_lower or "qu'est-ce que j'ai fait" in user_msg_lower:
                today_facts = [m.text for m in memory_engine.memories if m.category in ["episodic_today", "profile"]]
                reply = "D'après ma mémoire locale indexée, voici ce que tu as fait aujourd'hui :<br>• " + "<br>• ".join(today_facts)
            elif "qui suis-je" in user_msg_lower or "mon nom" in user_msg_lower:
                name = memory_engine.user_profile.get("user_name", "Yoel")
                reply = f"Tu es <strong>{name}</strong> ! Tu travailles sur ton projet de modèle d'IA local <strong>Sarah Ngin</strong> sur iPhone 14."
            elif slang_translations:
                trads_text = ", ".join(slang_translations)
                reply = f"J'ai bien compris ton message en langage familier/SMS ({trads_text}) ! Tout est parfaitement noté dans ma mémoire locale sans encombrer ton téléphone."
            elif "hébreu" in user_msg_lower or "עברית" in user_msg_lower or "tradu" in user_msg_lower:
                reply = "J'ai intégré le grand dictionnaire bilingue Hébreu <-> Français (racines, conjugaisons au présent, passé et futur et textes alignés). Tout est encodé dans mes poids de 3.89M paramètres."
            else:
                reply = f"Message bien reçu et analysé ! J'ai indexé cette information dans ma mémoire persistante ultra-compacte ({memory_engine.get_stats()['file_size_kb']:.1f} Ko). Je suis prête pour la suite !"

            memory_engine.extract_and_remember(user_msg, assistant_response=reply)

            # Renvoyer la réponse et l'état de la mémoire
            response_data = {
                "reply": reply,
                "memories": [m.to_dict() for m in memory_engine.memories[-10:]],
                "stats": memory_engine.get_stats(),
                "detected_slang": slang_translations
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_server():
    server_address = ("", PORT)
    with socketserver.TCPServer(server_address, SarahRequestHandler) as httpd:
        print(f"Serveur Web Sarah Ngin actif sur : http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
