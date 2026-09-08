# Sarah Ngin - Modèle d'Intelligence Artificielle Locale (iPhone 14)

**Sarah Ngin** est un modèle d'intelligence artificielle conversationnelle auto-régressif (Small Language Model - Transformer) développé intégralement de zéro (from scratch), optimisé pour les environnements mobiles et contraints en mémoire (< 4 Go RAM, calibré pour **Apple iPhone 14 / A15 Bionic**).

---

## 🚀 Test Immédiat du Modèle

Pour tester le modèle d'intelligence artificielle directement :

### 1. Test avec une question unique :
```bash
python test_sarah_model.py "Bonjour Sarah, comment vas-tu ?"
```

### 2. Mode conversation interactive en direct :
```bash
python test_sarah_model.py
```

### 3. Interface Web & Serveur Local :
```bash
python server.py
# Accès navigateur : http://localhost:8080
```

---

## 🏗️ Architecture Technique & Spécifications

* **Paramètres totaux :** `18,556,416` (18.55 M)
* **Vocabulaire :** `3,763` tokens (BPE Byte-level multilingue)
* **Dimension cachée (`d_model`) :** `384`
* **Couches (`layers`) :** `8`
* **Têtes d'attention (`heads`) :** `12` Q-Heads / `8` KV-Heads (Grouped-Query Attention)
* **Contexte maximal :** `128` tokens
* **Fonctionnalités :** RoPE (Rotary Position Embeddings), SwiGLU, RMSNorm, Weight Tying.

---

## 📦 Compilation & Exportation Mobile

Pour compiler et exporter le modèle dans les formats optimisés :

```bash
# Compilation JIT / TorchScript autonome & standalone package
python export/compile_model.py

# Quantification INT8 et FP16 pour Apple Neural Engine / Metal
python export/quantize_iphone14.py
```

### Empreinte mémoire sur iPhone 14 :
* **Modèle original FP32 :** ~70.8 Mo
* **Modèle exporté FP16 (Metal) :** ~35.4 Mo
* **Modèle quantifié INT8 :** ~23.3 Mo
* **RAM totale en fonctionnement :** < 65 Mo (< 2% de la RAM disponible sur iPhone 14)

---

## 📋 Rapport d'Audit & Évaluation

Le rapport d'audit exhaustif et transparent du modèle chargé est disponible dans :
* [AUDIT_COMPLET.md](AUDIT_COMPLET.md)
* [web/audit_report.html](web/audit_report.html) (Rapport visuel HTML)

---

## 📂 Structure du Répertoire

```
Local-LLM-Sarah/
├── agent_developer/          # Acquisition et nettoyage des données
├── checkpoints/              # Poids du modèle et tokenizers (8k)
├── data/                     # Datasets de conversation et corpus multilingue
├── export/                   # Compilation JIT, quantization FP16/INT8
│   ├── compile_model.py      # Compilation autonome
│   └── quantize_iphone14.py  # Quantification mobile
├── inference/                # Moteurs d'inférence autorégressifs
│   └── generate.py           # Générateur avec pénalité de répétition
├── memory/                   # Mémoire contextuelle persistante
├── model/                    # Définition de l'architecture Transformer & BitNet/MoE
├── tokenizer/                # Tokenizer BPE multilingue
├── training/                 # Pipelines d'entraînement et curriculum
├── web/                      # Interface web et tableau de bord
├── AUDIT_COMPLET.md          # Rapport d'audit complet
├── main.py                   # CLI unifiée
├── server.py                 # Serveur HTTP API & Web UI
└── test_sarah_model.py       # Script de test direct du modèle
```
