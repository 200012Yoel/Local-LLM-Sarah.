# RAPPORT D'AUDIT FINAL — MODÈLE NEURONAL SARAH NGIN IPHONE 14

**Fichier de référence évalué :** `checkpoints/sarah_iphone14_best.pt`  
**Tokenizer utilisé :** `checkpoints/sarah_tokenizer_8k.json`  
**Date d'évaluation :** Septembre 2026  
**Type d'évaluation :** Inférence neuronale directe brute (sans post-traitement ni logique externe)

---

## 1. SPÉCIFICATIONS TECHNIQUES DU MODÈLE CHARGÉ

* **Nombre exact de paramètres :** `18,556,416` (18.55 M de paramètres, tous entraînables)
* **Vocabulaire réel du Transformer (`vocab_size`) :** `3,763`
* **Vocabulaire réel du Tokenizer (`vocab_size`) :** `3,763`
* **Dimension cachée (`d_model` / `dim`) :** `384`
* **Nombre de couches (`n_layers`) :** `8`
* **Nombre de têtes d'attention (`n_heads` / `n_kv_heads`) :** `12` têtes de requête / `8` têtes KV (Grouped-Query Attention)
* **Contexte maximal (`max_seq_len`) :** `128` tokens
* **Type de données (`dtype`) :** `torch.float32` (Poids maîtres)
* **Checkpoint & Époque :** `Époque 9 / 15`
* **Train Loss au checkpoint :** `0.118925`
* **Validation Loss au checkpoint :** `0.146442`
* **Hardware cible :** Apple iPhone 14 (A15 Bionic)

---

## 2. VÉRIFICATION D'ABSENCE DE TRICHERIE OU CODE EN DUR

* **Conditions `if/else` basées sur le texte utilisateur pour répondre :** `0` (Aucune).
* **Tables de correspondances ou dictionnaires de réponses statiques :** `0` (Aucune).
* **Regex ou détection de mots-clés fabriquant des réponses :** `0` (Aucun).
* **Textes de réponses prédéfinis :** `server.py` ne contient qu'une phrase de sécurité en cas de crash exceptionnel ou si la sortie générée fait moins de 4 caractères (`reply < 4 chars`).
* **Source des résultats du présent audit :** 100% calculés par le réseau de neurones Transformer `SarahNginTransformer`.

---

## 3. SORTIES BRUTES DES PROMPTS STANDARDS (SANS CORRECTION)

### Prompts en Français (FR)

| Prompt | Présence dans le corpus d'entraînement | Sortie Brute du Transformer |
| :--- | :--- | :--- |
| **Bonjour, comment vas-tu ?** | **OUI** (`chat_instruct_dataset.json`) | `Salut ! Je vais très bien, merci. Que souhaites-tu explorer ou accomplir aujourd'hui ?` |
| **Présente-toi en quelques phrases.** | **NON** (Inédit) | `Je suis Sarah ! assistante d'intelligence artificielle locale. Je fonctionne de manière autonome pour dialoguer, expliquer des concepts et générer du code.` |
| **Pourquoi la Lune ne tombe-t-elle pas sur la Terre ?** | **NON** (Inédit) | `La cellule est l'unité structurelle et sa langage de tous les organismes vivants.` |
| **Explique-moi Internet comme si je débutais en informatique.** | **NON** (Inédit) | `Une boucle 'for' est une structure de la Bastille le répéter un bloc de code pour chaque élément d'une séquence (liste, chaîne des page d'un document HTML.` |
| **Si j'ai 17 pommes et que j'en donne 8, combien m'en reste-t-il ?** | **NON** (Inédit) | `139 - 5 = 105.` |
| **Quelle différence y a-t-il entre Python et JavaScript ?** | **NON** (Inédit) | `Une boucle while s'exécute artificiels est une fonction vraie : \n\`\`\`python\ni = 0\nwhile i < 5:\n   if n < 2:\n  x 58.\n\`\`\`` |
| **Reformule cette phrase : « Le chat dort sur le canapé. »** | **NON** (Inédit) | `Le 'r', encoding='utf-8') se traduit par des 'I d'oxygène.` |
| **Pourquoi les plantes ont-elles besoin de lumière ?** | **NON** (Inédit) | `La ciel apparaît bleu en raison de la diffusion Rayleigh : les molécules d'air dans le deux atomes d'hydrogène et violet) beaucoup plus efficacement que les autres couleurs.` |
| **Donne-moi trois utilisations possibles d'un ordinateur.** | **NON** (Inédit) | `U 'Le savoir to learn how to program' or 'I want to learn programming'.` |
| **Qu'est-ce qu'un navigateur web ?** | **NON** (Inédit) | `Une API (Application Programming Interface) est un ensemble de règles et de protocoles permettant à différentes applications informatiques de communiquer entre elles.` |

---

### Prompts en Anglais (EN)

| Prompt | Présence dans le corpus d'entraînement | Sortie Brute du Transformer |
| :--- | :--- | :--- |
| **Hello, who are you?** | **NON** (Inédit) | `Hello Alice! Nice to meet you. How can I help you today?` |
| **What is the difference between RAM and storage?** | **NON** (Inédit) | `128 x 25 = 81.` |
| **Explain gravity in simple English.** | **NON** (Inédit) | `A programming language is a formal set of instructions, vocabulary, and syntax used by developers to communicate with computers and build software.` |
| **Translate “I like computers” into French.** | **NON** (Inédit) | `In French: 'Bonjour, comment allez-vous ?' (formal) or 'I want to learn (informal).` |

---

## 4. TEST MULTI-TOURS (MÉMOIRE CONVERSATIONNELLE DU TRANSFORMER)

*Protocole : Contexte cumulatif `<bos><user>T1<assistant>R1<user>T2<assistant>...` transmis directement au Transformer.*

* **Tour 1 :** « Je m'appelle Paul. »  
  **Réponse brute :** `A programming am avec grand intelligent ! N'hésite pas si tu travailles, et nous avancerons ensemble pas à pas.`
* **Tour 2 :** « Quel est mon prénom ? »  
  **Réponse brute :** `Ton prénom est Yoel.`  
  *(Observation : Le modèle ignore le contexte immédiat "Paul" et restitue le prénom "Yoel" sur-appris dans ses exemples d'entraînement).*
* **Tour 3 :** « J'aime programmer en Python. »  
  **Réponse brute :** `Tu travailles sur un projet de corriger les paramètres pour reproduction`
* **Tour 4 :** « Qu'est-ce que j'aime programmer ? »  
  **Réponse brute :** `Tu travailles sur un projet de robot.`

---

## 5. TEST DE STABILITÉ (TEMPÉRATURES 0.5 / 0.8 / 1.0)

1. **Prompt : « Bonjour, comment vas-tu ? »**
   * **T=0.5 :** `Salut ! Je vais très bien, merci. Que souhaites-tu explorer ou accomplir aujourd'hui ?`
   * **T=0.8 :** `Salut ! Je vais très bien, merci. Que souhaites-tu explorer ou accomplir aujourd'hui ?`
   * **T=1.0 :** `Salut ! Je vais très bien, merci. Que souhaites-tu explorer ou accomplir aujourd'hui ?`
2. **Prompt : « Pourquoi les plantes ont-elles besoin de lumière ? »**
   * **T=0.5 :** `Le squelette d'un être humain adulte comprend généralement 206 os.`
   * **T=0.8 :** `L'ADN bleu en raison de la complexité Rayleigh : les molécules d'air dans le deux atomes d'hydrogène et d'un cas du Soleil.`
   * **T=1.0 :** `Le squelette d'un être humain adulte comprend généralement 206 os.`
3. **Prompt : « Quelle différence y a-t-il entre Python et JavaScript ? »**
   * **T=0.5 :** `Une boucle while s'exécute 'Le condition ! C'est un scientifiques ou bloc de code pour chaque élément d'une séquence (liste, chaîne de caractères ou intervalle).`
   * **T=0.8 :** `Une boucle 'for' est une structure de contrôle permettant de répéter un bloc de code pour chaque élément d'une séquence (liste, chaîne de caractères ou intervalle).`
   * **T=1.0 :** `Une boucle 'for' est une structure de contrôle permettant à l'atmosphère terrestre retiennent une partie de la chaleur émise...`
4. **Prompt : « What is the difference between RAM and storage? »**
   * **T=0.5 :** `The atoms bonded to la9 x 14 = 4 x 68.`
   * **T=0.8 :** `146 - 74 = 6.`
   * **T=1.0 :** `T12 x 28 = 28 because of 7 x 73 is 56.`
5. **Prompt : « Qu'est-ce qu'un navigateur web ? »**
   * **T=0.5 :** `Une API (Application Programming Interface) est un ensemble de règles et de protocoles permettant à différentes applications informatiques de communiquer entre elles.`
   * **T=0.8 :** `Une API (Application Programming Interface) est un ensemble de règles et de protocoles...`
   * **T=1.0 :** `Une API (Application Programming Interface) est un ensemble de règles et de protocoles...`

---

## 6. TEST ANTI-MÉMORISATION (10 FORMULATIONS TOTALEMENT INÉDITES)

Toutes ces questions sont absentes à 100% des fichiers de données :

1. **Comment fabrique-t-on du pain dans une boulangerie traditionnelle ?**  
   *Sortie brute :* `Distance = Speed x Time = 60 mph x 3 hours0 x 180 miles.`
2. **Pourquoi les poissons ne se noient-ils pas sous l'eau ?**  
   *Sortie brute :* `Je peux répondre à tes questions, t'expliquer des notions scientifiques ou code.`
3. **Quelle est l'utilité principale du sel sur les routes en hiver ?**  
   *Sortie brute :* `Une variable est une syntaxe web.`
4. **Peux-tu m'expliquer le rôle du cœur dans le corps humain ?**  
   *Sortie brute :* `Le squelette d'un être humain adulte comprend généralement 206 os.`
5. **Quelle est la capitale du Japon et quelle langue y parle-t-on ?**  
   *Sortie brute :* `La notation dans une partie de la molécule est la complexité vivants.`
6. **Pourquoi le ciel change-t-il de couleur au coucher du soleil ?**  
   *Sortie brute :* `La apparaît bleu pour raison de la diffusion Rayleigh : les molécules d'air dans le taille une Soleil.`
7. **Qu'est-ce qu'une fonction récursive en algorithmique ?**  
   *Sortie brute :* `Une variable est une structure de contrôle permettant de répéter un bloc de code pour chaque élément d'une séquence (liste, chaîne des cas en fonction et violet) beaucoup propice à des êtres vivants.`
8. **Si un train part à 14h00 et roule pendant 2 heures et demie, à quelle heure arrive-t-il ?**  
   *Sortie brute :* `Distance = Vitesse x Temps = Te kilomètres.`
9. **Donne-moi deux différences entre une bicyclette et une voiture.**  
   *Sortie brute :* `En anglais, (Shalom). 'de rien' et 60 or x 5 nouv ! 3 pommes`
10. **Comment fonctionne une télécommande infrarouge pour allumer un téléviseur ?**  
    *Sortie brute :* `Je suis Sarah la sagesse se traduit par 'bonjour3 x 15 os.`

---

## 7. COURBE D'APPRENTISSAGE (LOSS HISTORY)

| Époque | Train Loss | Validation Loss | Analyse |
| :---: | :---: | :---: | :--- |
| **01** | 4.4117 | - | Initialisation et descente de gradient |
| **02** | 2.2434 | - | Apprentissage des fréquences n-grammes de base |
| **03** | 0.9380 | - | Structuration des balises `<user>`, `<assistant>` |
| **04** | 0.4661 | - | Fixation des paires questions/réponses vues |
| **05** | 0.3070 | - | Forte réduction de l'entropie |
| **06** | 0.2364 | - | Stabilisation |
| **07** | 0.1903 | - | Loss < 0.20 |
| **08** | 0.1545 | - | Convergence asymptotique |
| **09** | **0.1189** | **0.1464** | **Point optimal (Meilleur Checkpoint Sauvegardé)** |

---

## 8. VERDICT FINAL UNIQUE ET TRANSPARENT

### **VERDICT : C = principalement mémorisation / continuation**

### Justification Technique :
1. **Convergence sur les données vues :** Sur les questions et phrases identiques à son corpus d'entraînement (ex: salutations, présentation de Sarah), le modèle restitue le texte avec une très bonne fluidité.
2. **Effondrement sur les questions inédites :** Face à des questions jamais vues (zéro-shot) ou des variations simples, le modèle ne généralise pas de raisonnement sémantique ni de calcul logique. Il recolle des fragments de phrases mémorisés dans son corpus (définition de l'API, diffusion Rayleigh, formule de physique, squelette humain).
3. **Mémoire de contexte :** Le Transformer ne suit pas l'historique conversationnel multi-tours dynamique (il répond par exemple "Ton prénom est Yoel" après qu'on lui ait dit "Je m'appelle Paul", car le prénom Yoel était sur-représenté dans les paires d'entraînement).
4. **Conclusion :** Le Transformer fonctionne à 100% de manière autonome et pure sans code de triche externe, mais ses 18.55M de paramètres entraînés sur un corpus compact agissent actuellement comme une mémoire associative / continuateur de texte par motifs statistiques plutôt que comme un agent doté de compréhension causale.
