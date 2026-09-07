"""
Script d'exécution et de validation de code dans le Sandbox pour Sarah Ngin.
"""

from testing_sandbox.code_verifier import CodeVerifierSandbox
from agent_developer.coding_corpus import CODING_SAMPLES

def run_tests():
    sandbox = CodeVerifierSandbox()
    samples = {
        "Python": "def compute_loss(y_pred, y_true):\n    return sum((p - t)**2 for p, t in zip(y_pred, y_true))\nresult = compute_loss([1.0, 2.0], [1.0, 2.0])\nassert result == 0.0",
        "JavaScript": "function formatMemory(bytes) { return (bytes / (1024 * 1024)).toFixed(2) + ' MB'; }\nconsole.log(formatMemory(157286400));",
        "HTML & CSS": "<!DOCTYPE html><html><head><style>.card { color: blue; }</style></head><body><div class='card'>Sarah Ngin</div></body></html>",
        "Java": "public class TestModel { public static void main(String[] args) { System.out.println(\"Sarah Ngin OK\"); } }",
        "Swift": "import SwiftUI\nstruct ContentView: View { var body: some View { Text(\"Sarah Ngin\") } }"
    }

    report = sandbox.run_full_suite(samples)
    print("="*60)
    print("      RAPPORT DE VALIDATION DU CODE - SARAH NGIN")
    print("="*60)
    print(f"Total Tests : {report['total_tested']}")
    print(f"Réussis     : {report['passed']}")
    print(f"Échoués     : {report['failed']}")
    print(f"Taux Succès : {report['success_rate_pct']}%\n")
    for lang, res in report["results_by_language"].items():
        print(f"[{res['status']}] {lang:<15} : {res['details']}")
    print("="*60)

if __name__ == "__main__":
    run_tests()
