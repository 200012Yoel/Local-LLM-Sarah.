# Sarah Ngin - Modèle d'Intelligence Artificielle 100% From Scratch

**Sarah Ngin** est un modèle d'intelligence artificielle auto-régressif (Small Language Model - Transformer) développé intégralement de zéro (sans dépendances pré-entraînées), optimisé pour tourner sur des environnements contraints en mémoire (< 4 Go RAM, parfait pour **iPhone 14**).

---

## 🏗️ Architecture Technique

```
Sarah_Ngin/
├── agent_developer/          # Agent Développeur autonome (acquisition & nettoyage)
│   ├── data_collector.py     # Dictionnaires FR/HE/EN/ZH et corpus logiques
│   └── cleaner.py            # Normalisation Unicode NFC et filtrage
├── tokenizer/                # Tokenizer BPE (Byte-Pair Encoding) de zéro
│   └── bpe_tokenizer.py      # Zéro OOV (Unicode Byte-level fallback multilingue)
├── model/                    # Réseau de Neurones Transformer
│   ├── config.py             # Hyperparamètres modulaires (Mobile / Standard)
│   └── transformer.py        # RMSNorm, RoPE (Rotary Embeddings), SwiGLU, Weight Tying
├── training/                 # Pipeline d'Entraînement
│   ├── dataset.py            # Dataloader à fenêtrage glissant
│   └── trainer.py            # AdamW, Cosine Warmup, Gradient Clipping, Perplexité
├── inference/                # Moteur de Génération
│   └── generate.py           # Top-K, Top-P (Nucleus), Température, Pénalité de répétition
├── export/                   # Export Mobile & Edge
│   └── export_mobile.py      # Quantification INT8, TorchScript iOS, ONNX, Benchmarks
├── checkpoints/              # Poids du modèle et vocabulaire
├── mobile_build/             # Modèles exportés pour iPhone 14
└── main.py                   # Interface CLI unifiée
```

---

## ⚡ Caractéristiques pour iPhone 14 (4 Go RAM)

- **Paramètres :** ~3.6M à 25M paramètres (selon configuration)
- **Empreinte mémoire :** 
  - **FP32 :** ~13.8 Mo
  - **Quantifié INT8 :** ~3.46 Mo
  - **RAM en fonctionnement :** < 100 Mo (Consomme moins de 2.5% des 4 Go de RAM d'un iPhone 14)
- **Latence :** Inférence ultra-rapide sur CPU / Neural Engine (A15 Bionic).

---

## 🚀 Utilisation Rapide

### 1. Exécuter le pipeline complet (Collecte -> Tokenizer -> Entraînement -> Test -> Export Mobile)
```bash
python main.py --action all --epochs 25
```

### 2. Discuter en direct avec Sarah Ngin (Mode Chat Interactif)
```bash
python main.py --action chat
```

### 3. Exécuter des étapes spécifiques
- **Collecte des dictionnaires par l'Agent Développeur :**
  ```bash
  python main.py --action collect
  ```
- **Entraînement du Tokenizer BPE :**
  ```bash
  python main.py --action tokenize
  ```
- **Entraînement du réseau de neurones :**
  ```bash
  python main.py --action train --epochs 30
  ```
- **Test de génération textuelle :**
  ```bash
  python main.py --action generate
  ```
- **Exportation pour iPhone / Mobile :**
  ```bash
  python main.py --action export
  ```

---

## 🌐 Langues & Logique
- **Français :** Définitions complètes, grammaire, relations sujet-verbe-complément, syllogismes logiques.
- **Hébreu (עברית) :** Vocabulaire, concepts fondamentaux, alignement sémantique.
- **Anglais (English) :** Termes techniques, structure des phrases, correspondances.
- **Chinois (中文) :** Caractères essentiels, traductions, paires bilingues.
