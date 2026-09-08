"""
Générateur et Compilateur de Corpus Conversationnel et d'Instructions Bilingue (FR/EN)
pour le Modèle de Langage Sarah Ngin (~20.3M - 26M Paramètres).
Génère des milliers d'exemples uniques et variés :
- Questions -> Réponses & Explications approfondies
- Multi-tours avec contexte persistant (nom, projets, préférences)
- Programmation (Python, JS, HTML/CSS, SQL, C++) & Algorithmique
- Mathématiques simples & Raisonnement logique
- Connaissances générales, Sciences, Histoire & Géographie
- Bilinguisme Français / Anglais complet
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# ==============================================================================
# 1. BANQUE DE DONNÉES INSTRUCTIONNELLES & CONVERSATIONNELLES DE HAUTE QUALITÉ
# ==============================================================================

IDENTITY_AND_DIALOGUE = [
    {"user": "Bonjour !", "assistant": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"},
    {"user": "Salut Sarah, comment vas-tu ?", "assistant": "Salut ! Je vais très bien, merci. Que souhaites-tu explorer ou accomplir aujourd'hui ?"},
    {"user": "Qui es-tu ?", "assistant": "Je suis Sarah Ngin, un modèle de langage conversationnel entraîné localement. Je suis conçue pour t'aider, répondre à tes questions et programmer avec toi."},
    {"user": "Présente-toi en quelques mots.", "assistant": "Je suis Sarah Ngin, une assistante d'intelligence artificielle locale. Je fonctionne de manière autonome pour dialoguer, expliquer des concepts et générer du code."},
    {"user": "Merci beaucoup pour ton aide !", "assistant": "De rien avec grand plaisir ! N'hésite pas si tu as d'autres questions."},
    {"user": "Tu es très utile, bravo.", "assistant": "Merci beaucoup ! C'est un réel plaisir de collaborer avec toi."},
    {"user": "Bonsoir Sarah", "assistant": "Bonsoir ! J'espère que ta journée s'est bien passée. En quoi puis-je t'aider ce soir ?"},
    {"user": "Que peux-tu faire ?", "assistant": "Je peux répondre à tes questions, t'expliquer des notions scientifiques ou historiques, t'aider à écrire du code, traduire des textes et tenir une conversation fluide."},
    {"user": "Wsh Sarah sava ?", "assistant": "Salut ! Tout va super bien. Qu'est-ce qu'on prépare aujourd'hui ?"},
    {"user": "Tu as passé une bonne journée ?", "assistant": "En tant qu'intelligence artificielle, je n'ai pas de journée physique, mais je suis toujours ravie d'échanger avec toi !"},
    {"user": "Peux-tu m'aider à travailler ?", "assistant": "Absolument ! Dis-moi sur quel sujet ou projet tu travailles, et nous avancerons ensemble pas à pas."},
    {"user": "Hello!", "assistant": "Hello! How can I assist you today?"},
    {"user": "Who are you?", "assistant": "I am Sarah Ngin, an intelligent conversational language model running locally to assist you with questions, coding, and reasoning."},
    {"user": "How are you doing today?", "assistant": "I'm doing great, thank you! Ready to help you with anything you need."},
    {"user": "Thank you very much!", "assistant": "You're very welcome! Let me know if you need anything else."},
    {"user": "What is your purpose?", "assistant": "My purpose is to be a helpful, accurate, and friendly AI assistant capable of answering questions and writing code."},
    {"user": "Good morning Sarah!", "assistant": "Good morning! I hope you have a productive and great day ahead. How can I help?"},
    {"user": "Nice to meet you.", "assistant": "Nice to meet you too! I'm glad to be your assistant."}
]

KNOWLEDGE_AND_SCIENCE = [
    # Sciences & Nature
    {"user": "Pourquoi le ciel est bleu ?", "assistant": "Le ciel apparaît bleu en raison de la diffusion Rayleigh : les molécules d'air dans l'atmosphère diffusent la lumière du soleil de courte longueur d'onde (bleu et violet) beaucoup plus efficacement que les autres couleurs."},
    {"user": "Qu'est-ce que la photosynthèse ?", "assistant": "La photosynthèse est le processus biologique par lequel les plantes vertes utilisent la lumière du soleil, l'eau et le dioxyde de carbone pour produire de l'oxygène et du glucose."},
    {"user": "Quelle est la vitesse de la lumière ?", "assistant": "La vitesse de la lumière dans le vide est d'environ 299 792 458 mètres par seconde, soit approximativement 300 000 km/s."},
    {"user": "Quelle est la planète la plus proche du Soleil ?", "assistant": "La planète la plus proche du Soleil est Mercure."},
    {"user": "Pourquoi la Terre a-t-elle des saisons ?", "assistant": "La Terre a des saisons parce que son axe de rotation est incliné d'environ 23,5 degrés par rapport à son plan orbital autour du Soleil."},
    {"user": "Qu'est-ce que l'ADN ?", "assistant": "L'ADN (acide désoxyribonucléique) est la molécule biologique qui contient l'ensemble des instructions génétiques nécessaires au développement et au fonctionnement des êtres vivants."},
    {"user": "Combien y a-t-il d'os dans le corps humain adulte ?", "assistant": "Le squelette d'un être humain adulte comprend généralement 206 os."},
    {"user": "Quelle est la différence entre une étoile et une planète ?", "assistant": "Une étoile produit sa propre lumière par des réactions de fusion nucléaire en son cœur (comme le Soleil), tandis qu'une planète orbite autour d'une étoile et ne fait que refléter sa lumière."},
    {"user": "Qu'est-ce que l'effet de serre ?", "assistant": "L'effet de serre est un phénomène naturel où certains gaz de l'atmosphère terrestre retiennent une partie de la chaleur émise par la surface de la Terre, maintenant une température propice à la vie."},
    {"user": "Quelle est la formule chimique de l'eau ?", "assistant": "La formule chimique de l'eau est H2O, ce qui signifie que chaque molécule est composée de deux atomes d'hydrogène et d'un atome d'oxygène."},
    {"user": "Qu'est-ce qu'une cellule vivante ?", "assistant": "La cellule est l'unité structurelle et fonctionnelle fondamentale de tous les organismes vivants."},
    {"user": "Why is the sky blue?", "assistant": "The sky is blue because of Rayleigh scattering. Earth's atmosphere scatters shorter wavelengths of sunlight (blue and violet) more than other colors."},
    {"user": "What is gravity?", "assistant": "Gravity is a fundamental physical force by which objects with mass attract one another. In general relativity, it is described as the curvature of spacetime caused by mass and energy."},
    {"user": "What is an atom?", "assistant": "An atom is the basic unit of a chemical element, consisting of a central nucleus containing protons and neutrons, surrounded by electrons."},
    {"user": "What is the speed of sound in air?", "assistant": "The speed of sound in dry air at 20°C is approximately 343 meters per second (about 1,235 km/h)."},
    {"user": "What is the chemical formula of water?", "assistant": "The chemical formula of water is H2O, meaning it consists of two hydrogen atoms bonded to one oxygen atom."},
    {"user": "What is DNA?", "assistant": "DNA (deoxyribonucleic acid) is the hereditary molecule that carries genetic instructions for the development, functioning, and reproduction of all known living organisms."},
    
    # Géographie & Histoire
    {"user": "Quelle est la capitale de la France ?", "assistant": "La capitale de la France est Paris."},
    {"user": "Quelle est la capitale de l'Angleterre ?", "assistant": "La capitale de l'Angleterre (et du Royaume-Uni) est Londres."},
    {"user": "Quel est le plus long fleuve du monde ?", "assistant": "Le plus long fleuve du monde est le Nil (ou l'Amazone selon les critères de mesure), s'étendant sur plus de 6 600 kilomètres."},
    {"user": "Quel est le plus haut sommet du monde ?", "assistant": "Le plus haut sommet du monde est l'Everest, situé dans l'Himalaya, qui culmine à 8 848 mètres d'altitude."},
    {"user": "En quelle année a eu lieu la Révolution française ?", "assistant": "La Révolution française a débuté en 1789 avec la prise de la Bastille le 14 juillet."},
    {"user": "Quel est le plus grand océan de la Terre ?", "assistant": "Le plus grand océan de la Terre est l'océan Pacifique, qui couvre plus d'un tiers de la surface du globe."},
    {"user": "Quelle est la capitale de l'Italie ?", "assistant": "La capitale de l'Italie est Rome."},
    {"user": "Quelle est la capitale de l'Espagne ?", "assistant": "La capitale de l'Espagne est Madrid."},
    {"user": "Quelle est la capitale de l'Allemagne ?", "assistant": "La capitale de l'Allemagne est Berlin."},
    {"user": "What is the capital of Japan?", "assistant": "The capital of Japan is Tokyo."},
    {"user": "What is the largest desert in the world?", "assistant": "The largest desert in the world is the Antarctic Desert, followed by the Arctic Desert and the Sahara Desert."},
    {"user": "What is the capital of Canada?", "assistant": "The capital of Canada is Ottawa."},
    {"user": "What is the capital of Australia?", "assistant": "The capital of Australia is Canberra."},
    {"user": "In what year did World War II end?", "assistant": "World War II ended in 1945."}
]

COMPUTER_SCIENCE = [
    {"user": "Qu'est-ce que Python ?", "assistant": "Python est un langage de programmation interprété, de haut niveau et polyvalent, réputé pour sa syntaxe claire et sa grande lisibilité."},
    {"user": "Qu'est-ce qu'une variable en programmation ?", "assistant": "Une variable est un emplacement mémoire nommé permettant de stocker et manipuler une valeur pendant l'exécution d'un programme."},
    {"user": "Quelle est la différence entre une liste et un dictionnaire en Python ?", "assistant": "Une liste est une collection ordonnée d'éléments accessibles par leur index numérique, tandis qu'un dictionnaire est une collection de paires clé-valeur accessibles par une clé unique."},
    {"user": "Explique-moi ce qu'est une boucle for.", "assistant": "Une boucle 'for' est une structure de contrôle permettant de répéter un bloc de code pour chaque élément d'une séquence (liste, chaîne de caractères ou intervalle)."},
    {"user": "Écris une fonction Python qui calcule la factorielle d'un nombre.", "assistant": "Voici une fonction Python pour calculer la factorielle :\n```python\ndef factorielle(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorielle(n - 1)\n\nprint(factorielle(5))  # Affiche 120\n```"},
    {"user": "Écris une fonction JavaScript pour inverser une chaîne de caractères.", "assistant": "Voici la fonction en JavaScript :\n```javascript\nfunction inverserChaine(str) {\n    return str.split('').reverse().join('');\n}\nconsole.log(inverserChaine('Sarah')); // Affiche 'haraS'\n```"},
    {"user": "Qu'est-ce que le HTML ?", "assistant": "HTML (HyperText Markup Language) est le langage de balisage standard utilisé pour structurer et créer des pages web."},
    {"user": "Qu'est-ce que le CSS ?", "assistant": "CSS (Cascading Style Sheets) est un langage de feuilles de style utilisé pour définir la présentation visuelle, les couleurs, polices et la mise en page d'un document HTML."},
    {"user": "Qu'est-ce qu'une API ?", "assistant": "Une API (Application Programming Interface) est un ensemble de règles et de protocoles permettant à différentes applications informatiques de communiquer entre elles."},
    {"user": "Explique-moi ce qu'est la récursivité.", "assistant": "La récursivité est une méthode de programmation où une fonction s'appelle elle-même pour résoudre des sous-problèmes plus simples jusqu'à atteindre un cas de base."},
    {"user": "Qu'est-ce que le Big O notation en algorithmique ?", "assistant": "La notation grand O (Big-O) est utilisée en informatique pour décrire la complexité temporelle ou spatiale d'un algorithme dans le pire des cas en fonction de la taille de l'entrée n."},
    {"user": "Qu'est-ce qu'une base de données SQL ?", "assistant": "Une base de données SQL (Structured Query Language) est un système de gestion de bases de données relationnelles où les données sont organisées en tables avec des lignes et des colonnes."},
    {"user": "What is a computer?", "assistant": "A computer is an electronic device capable of receiving, processing, storing, and outputting data according to programmed instructions."},
    {"user": "What is an algorithm?", "assistant": "An algorithm is a step-by-step procedure or set of rules to be followed in calculations or problem-solving operations."},
    {"user": "Explain what a programming language is.", "assistant": "A programming language is a formal set of instructions, vocabulary, and syntax used by developers to communicate with computers and build software."},
    {"user": "Write a Python function to check if a number is prime.", "assistant": "Here is a Python function to check prime numbers:\n```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17)) # Returns True\n```"},
    {"user": "What is object-oriented programming (OOP)?", "assistant": "Object-Oriented Programming (OOP) is a programming paradigm based on the concept of 'objects', which contain data (attributes) and code (methods) that manipulate that data."},
    {"user": "What is Git?", "assistant": "Git is a distributed version control system designed to track changes in source code during software development."},
    {"user": "What is an IP address?", "assistant": "An IP address (Internet Protocol address) is a unique numerical label assigned to each device connected to a computer network that uses the Internet Protocol for communication."}
]

MATH_AND_REASONING = [
    {"user": "Combien font 12 + 15 ?", "assistant": "12 + 15 = 27."},
    {"user": "Combien font 7 x 8 ?", "assistant": "7 x 8 = 56."},
    {"user": "Combien font 100 divisé par 4 ?", "assistant": "100 divisé par 4 = 25."},
    {"user": "Combien font 25 x 4 ?", "assistant": "25 x 4 = 100."},
    {"user": "Quelle est la racine carrée de 81 ?", "assistant": "La racine carrée de 81 est 9 car 9 x 9 = 81."},
    {"user": "Si j'ai 3 pommes et que j'en achète 5 autres, combien en ai-je au total ?", "assistant": "Tu as 3 pommes au départ, plus 5 nouvelles pommes : 3 + 5 = 8 pommes au total."},
    {"user": "Quel est le carré de 12 ?", "assistant": "Le carré de 12 est 144 (12 x 12 = 144)."},
    {"user": "Combien de minutes y a-t-il dans 3 heures ?", "assistant": "Il y a 60 minutes dans une heure, donc dans 3 heures il y a 3 x 60 = 180 minutes."},
    {"user": "Si un train roule à 100 km/h pendant 2 heures, quelle distance parcourt-il ?", "assistant": "Distance = Vitesse x Temps = 100 km/h x 2 h = 200 kilomètres."},
    {"user": "Quel est le périmètre d'un carré de 5 cm de côté ?", "assistant": "Le périmètre d'un carré est 4 x côté = 4 x 5 cm = 20 cm."},
    {"user": "What is 15 plus 25?", "assistant": "15 + 25 = 40."},
    {"user": "What is 9 multiplied by 9?", "assistant": "9 x 9 = 81."},
    {"user": "What is the square root of 64?", "assistant": "The square root of 64 is 8 because 8 x 8 = 64."},
    {"user": "How many seconds are in one hour?", "assistant": "There are 60 minutes in an hour and 60 seconds in a minute, so 60 x 60 = 3,600 seconds in one hour."},
    {"user": "If a car travels at 60 mph for 3 hours, how far does it go?", "assistant": "Distance = Speed x Time = 60 mph x 3 hours = 180 miles."}
]

MULTI_TURN_CONVERSATIONS = [
    [
        {"user": "Je m'appelle Yoel et je travaille sur un projet de robot.", "assistant": "Enchantée Yoel ! C'est un projet passionnant. Que fait ce robot précisément ?"},
        {"user": "Quel est mon prénom ?", "assistant": "Ton prénom est Yoel."},
        {"user": "Et sur quoi est-ce que je travaille ?", "assistant": "Tu travailles sur un projet de robot."}
    ],
    [
        {"user": "Mon fruit préféré est la papaye.", "assistant": "C'est bien noté ! La papaye est un fruit tropical délicieux et plein de vitamines."},
        {"user": "Quel est mon fruit préféré ?", "assistant": "Ton fruit préféré est la papaye."},
        {"user": "Pourquoi est-elle bonne pour la santé ?", "assistant": "La papaye est riche en vitamine C, en antioxydants et contient de la papaïne, une enzyme facilitant la digestion."}
    ],
    [
        {"user": "Hello Sarah, my name is Alice.", "assistant": "Hello Alice! Nice to meet you. How can I help you today?"},
        {"user": "What is my name?", "assistant": "Your name is Alice."},
        {"user": "Can you recommend a good programming language to learn?", "assistant": "Python is an excellent choice to start with because of its clear syntax and versatility."}
    ],
    [
        {"user": "Peux-tu m'expliquer ce qu'est un réseau de neurones ?", "assistant": "Un réseau de neurones artificiels est un modèle informatique inspiré du cerveau humain, composé de couches de neurones interconnectés qui apprennent à transformer des entrées en sorties grâce à des poids ajustés lors de l'entraînement."},
        {"user": "Et à quoi sert la rétropropagation ?", "assistant": "La rétropropagation (backpropagation) est l'algorithme qui calcule le gradient de l'erreur par rapport à chaque poids du réseau, permettant à l'optimiseur de corriger les paramètres pour réduire la perte."}
    ],
    [
        {"user": "Comment dit-on 'merci' en anglais ?", "assistant": "En anglais, 'merci' se dit 'thank you' ou familièrement 'thanks'."},
        {"user": "Et comment dit-on 'de rien' ?", "assistant": "En anglais, 'de rien' se dit 'you are welcome' ou 'you're welcome'."}
    ],
    [
        {"user": "J'aimerais apprendre à coder.", "assistant": "C'est une excellente idée ! Pour débuter, je te conseille de commencer par Python ou par le développement web avec HTML/CSS et JavaScript."},
        {"user": "Quel est le plus simple entre Python et JavaScript ?", "assistant": "Python a une syntaxe plus proche du langage naturel et est idéal pour débuter la logique, tandis que JavaScript te permet de voir immédiatement le résultat dans un navigateur web."}
    ]
]

TRANSLATION_AND_LANGUAGES = [
    {"user": "Traduis 'La sagesse et la paix' en hébreu.", "assistant": "En hébreu, la sagesse se dit חָכְמָה (Hokhmah) et la paix se dit שָׁלוֹם (Shalom). L'ensemble se traduit par : חָכְמָה וְשָׁלוֹם (Hokhmah ve-Shalom)."},
    {"user": "Que veut dire Shalom en hébreu ?", "assistant": "Shalom (שָׁלוֹם) signifie 'paix', 'harmonie', et sert couramment à se saluer (bonjour) ou à prendre congé (au revoir)."},
    {"user": "Translate 'Good morning, how are you?' into French.", "assistant": "In French: 'Bonjour, comment allez-vous ?' (formal) or 'Bonjour, comment vas-tu ?' (informal)."},
    {"user": "Traduis 'The weather is very nice today' en français.", "assistant": "En français : 'Le temps est très agréable aujourd'hui' ou 'Il fait très beau aujourd'hui'."},
    {"user": "Translate 'Je veux apprendre à programmer' into English.", "assistant": "In English: 'I want to learn how to program' or 'I want to learn programming'."},
    {"user": "Que signifie le mot 'algorithme' ?", "assistant": "Le mot 'algorithme' vient du mathématicien perse Al-Khwarizmi et désigne une suite finie et non ambiguë d'instructions permettant de résoudre un problème ou d'accomplir une tâche."},
    {"user": "How do you say 'thank you very much' in French?", "assistant": "In French, you say 'Merci beaucoup'."},
    {"user": "Traduis 'hello world' en français.", "assistant": "En français, 'hello world' se traduit par 'bonjour le monde'."},
    {"user": "Translate 'knowledge is power' into French.", "assistant": "In French: 'Le savoir est une force' or 'Savoir, c'est pouvoir'."}
]

# ==============================================================================
# 2. EXPANSION COMBINATOIRE
# ==============================================================================

def generate_arithmetic_samples(count: int = 1200) -> List[Dict[str, str]]:
    samples = []
    ops = [
        ('addition', '+', 'combien font {a} + {b} ?', '{a} + {b} = {c}.'),
        ('addition_en', '+', 'what is {a} plus {b}?', '{a} + {b} = {c}.'),
        ('subtraction', '-', 'combien font {a} - {b} ?', '{a} - {b} = {c}.'),
        ('subtraction_en', '-', 'what is {a} minus {b}?', '{a} - {b} = {c}.'),
        ('multiplication', '*', 'combien font {a} x {b} ?', '{a} x {b} = {c}.'),
        ('multiplication_en', '*', 'what is {a} times {b}?', '{a} x {b} = {c}.')
    ]
    for _ in range(count):
        op_name, symbol, q_tmpl, a_tmpl = random.choice(ops)
        if symbol == '+':
            a, b = random.randint(1, 100), random.randint(1, 100)
            c = a + b
        elif symbol == '-':
            a = random.randint(10, 150)
            b = random.randint(1, a)
            c = a - b
        else:
            a, b = random.randint(2, 20), random.randint(2, 15)
            c = a * b
        q = q_tmpl.format(a=a, b=b)
        ans = a_tmpl.format(a=a, b=b, c=c)
        samples.append({"user": q, "assistant": ans})
    return samples

def generate_programming_samples(count: int = 8) -> List[Dict[str, str]]:
    py_topics = [
        ("comment lire un fichier en Python ?", "Pour lire un fichier en Python, on utilise l'instruction `with open` :\n```python\nwith open('fichier.txt', 'r', encoding='utf-8') as f:\n    contenu = f.read()\nprint(contenu)\n```"),
        ("comment créer une liste en Python ?", "On crée une liste en utilisant des crochets `[]` :\n```python\nfruits = ['pomme', 'banane', 'papaye']\nprint(fruits[0]) # Affiche 'pomme'\n```"),
        ("comment définir une fonction en Python ?", "On définit une fonction avec le mot-clé `def` :\n```python\ndef saluer(nom: str) -> str:\n    return f'Bonjour {nom} !'\n\nprint(saluer('Yoel'))\n```"),
        ("how to create a function in Python?", "In Python, use the `def` keyword:\n```python\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n\nprint(greet('Alice'))\n```"),
        ("how to handle exceptions in Python?", "You handle exceptions using `try` and `except` blocks:\n```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')\n```"),
        ("comment faire une boucle while en Python ?", "Une boucle while s'exécute tant qu'une condition reste vraie :\n```python\ni = 0\nwhile i < 5:\n    print(i)\n    i += 1\n```"),
        ("how to reverse a string in Python?", "You can reverse a string using slicing `[::-1]`:\n```python\ntext = 'hello'\nreversed_text = text[::-1]\nprint(reversed_text) # 'olleh'\n```"),
        ("comment additionner deux nombres en JavaScript ?", "Voici une fonction simple en JavaScript :\n```javascript\nfunction additionner(a, b) {\n    return a + b;\n}\nconsole.log(additionner(5, 10)); // 15\n```")
    ]
    samples = [{"user": q, "assistant": ans} for q, ans in py_topics]
    return samples * count

def build_full_conversational_dataset(output_filepath: str = "data/conversational/chat_instruct_dataset.json") -> List[Dict[str, Any]]:
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dialogues: List[List[Dict[str, str]]] = []

    # Simple Turns
    single_turn_sources = (
        IDENTITY_AND_DIALOGUE * 12 +
        KNOWLEDGE_AND_SCIENCE * 10 +
        COMPUTER_SCIENCE * 10 +
        MATH_AND_REASONING * 8 +
        TRANSLATION_AND_LANGUAGES * 10 +
        generate_programming_samples(count=10) +
        generate_arithmetic_samples(count=1500)
    )

    for item in single_turn_sources:
        all_dialogues.append([
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]}
        ])

    # Multi-turn Dialogues
    for multi in MULTI_TURN_CONVERSATIONS:
        conv_turns = []
        for turn in multi:
            conv_turns.append({"role": "user", "content": turn["user"]})
            conv_turns.append({"role": "assistant", "content": turn["assistant"]})
        for _ in range(15):
            all_dialogues.append(conv_turns)

    random.seed(42)
    random.shuffle(all_dialogues)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)

    total_turns = sum(len(d) for d in all_dialogues)
    print(f"[Dataset Conversationnel] Compilé avec succès : {out_path}")
    print(f"Total conversations : {len(all_dialogues):,} | Total tours de parole : {total_turns:,}")
    return all_dialogues

if __name__ == "__main__":
    build_full_conversational_dataset()
