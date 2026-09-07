"""
Dictionnaire Encyclopédique d'Argot Français, Verlan et Langage Texto / SMS.
Permet à Sarah Ngin de comprendre à 100% le langage parlé des jeunes et les abréviations de SMS.
"""

from typing import Dict, List, Tuple

# Base de données exhaustive de l'argot contemporain et du langage SMS
ARGOT_SMS_DICTIONARY: Dict[str, Tuple[str, str, str]] = {
    # Acronymes & SMS
    "tkt": ("t'inquiète", "Ne t'inquiète pas / Sois rassuré", "Tkt je gère la situation."),
    "tkt pas": ("ne t'inquiète pas", "Rassure-toi", "Tkt pas, tout est sous contrôle."),
    "jpp": ("j'en peux plus", "Expression d'épuisement ou de rire incontrôlable", "Jpp de cette blague elle est trop drôle !"),
    "mdr": ("mort de rire", "Expression pour indiquer que quelque chose est très drôle", "Mdr c'est exactement ça."),
    "ptdr": ("pété de rire", "Rire aux éclats / Hilarité intense", "Ptdr regarde ce qu'il vient de faire."),
    "xpdr": ("explosé de rire", "Rire extrême", "Xpdr c'est incroyable."),
    "askip": ("à ce qu'il paraît", "Selon les rumeurs / D'après ce qu'on dit", "Askip Sarah Ngin est super rapide sur iPhone 14."),
    "oklm": ("au calme", "Tranquillement, sans stress, en toute sérénité", "Je révise oklm dans mon salon."),
    "wsh": ("wesh / salut / quoi", "Salutation familière ou marque d'étonnement", "Wsh bien ou quoi ?"),
    "wesh": ("salut / hé / quoi", "Interjection familière pour saluer ou interpeller", "Wesh l'équipe, comment ça va ?"),
    "sah": ("en vrai / sérieusement", "Pour de vrai, en toute sincérité (de l'arabe)", "Sah, ce modèle d'intelligence artificielle est impressionnant."),
    "sah quel plaisir": ("quel réel plaisir", "Expression de satisfaction sincère", "Sah quel plaisir de voir le modèle fonctionner."),
    "bg": ("beau gosse / belle gosse", "Personne séduisante ou remarquable", "Bien joué bg, super travail."),
    "frero": ("frérot / mon frère", "Terme d'affection pour un ami proche", "Merci frero pour ton aide."),
    "frerot": ("frérot / ami très proche", "Ami intime / personne de confiance", "Tranquille frérot, on avance."),
    "dcp": ("du coup", "Par conséquent / Donc", "Dcp on fait quoi maintenant ?"),
    "stp": ("s'il te plaît", "Formule de politesse", "Envoie-moi le fichier stp."),
    "svp": ("s'il vous plaît", "Formule de politesse formelle", "Pouvez-vous confirmer svp ?"),
    "bcp": ("beaucoup", "En grande quantité", "Il y a bcp de données dans le dictionnaire."),
    "tt": ("tout", "Totalité", "C'est tt bon pour moi."),
    "pk": ("pourquoi", "Interrogation sur la raison", "Pk tu dis ça ?"),
    "prk": ("pourquoi", "Demande d'explication", "Prk le modèle est si rapide ?"),
    "pr": ("pour", "Préposition de destination ou but", "C'est un modèle pr mobile."),
    "vs": ("vous", "Pronom vous", "Vs êtes prêts ?"),
    "ns": ("nous", "Pronom nous", "Ns allons tester."),
    "cimer": ("merci (verlan)", "Remerciement chaleureux", "Cimer pour les explications."),
    "merci bcp": ("merci beaucoup", "Remerciement appuyé", "Merci bcp pour ton aide."),
    "a+": ("à plus tard", "Salutation de départ", "A+ et bonne journée."),
    "a toute": ("à tout à l'heure", "Formule pour signifier qu'on se revoit bientôt", "Je mange et je reviens, à toute."),
    "bjr": ("bonjour", "Salutation en début de journée", "Bjr tout le monde."),
    "bsr": ("bonsoir", "Salutation en soirée", "Bsr, comment s'est passée la journée ?"),
    "slm": ("salam / salut", "Salutation bienveillante", "Slm mon ami."),
    
    # Argot, Verlan & Vocabulaire Urbain
    "chelou": ("louche / bizarre", "Étrange, suspect ou hors du commun", "Ce comportement est un peu chelou."),
    "relou": ("lourd / pénible", "Ennuyeux, difficile ou agaçant", "C'est relou quand ça plante, mais là tout marche."),
    "zarbi": ("bizarre", "Étrange / singulier", "C'est un truc zarbi mais efficace."),
    "vénère": ("énervé (verlan)", "En colère ou irrité", "Pas besoin d'être vénère, le modèle s'adapte."),
    "chanmé": ("méchant / super bien (verlan)", "Formidable, génial, extraordinaire", "Cette architecture IA est trop chanmé."),
    "ouf": ("fou (verlan)", "Incroyable, exceptionnel ou démesuré", "C'est un truc de ouf cette vitesse."),
    "kiffer": ("aimer / apprécier énormément", "Prendre du plaisir à quelque chose", "Je kiffe utiliser Sarah Ngin."),
    "seum": ("amertume / déception", "Avoir la rage ou être dégoûté", "Il a le seum d'avoir raté."),
    "avoir le seum": ("être dégoûté", "Ressentir une vive frustration", "J'ai pas le seum, tout fonctionne."),
    "masterclass": ("coup de génie / réussite parfaite", "Œuvre ou action de très haut niveau", "Ce code c'est une vraie masterclass."),
    "carré": ("parfait / net et précis", "Tout est réglé, en ordre et conforme", "Tout est carré mon pote."),
    "c'est carré": ("c'est parfait / validé", "Tout est en ordre irréprochable", "C'est carré dans l'axe."),
    "mif": ("la famille (verlan)", "Les proches, la famille", "Passe le bonjour à la mif."),
    "poto": ("pote / ami", "Camarade ou ami fidèle", "Quoi de neuf poto ?"),
    "banger": ("un chef-d'œuvre musical ou technique", "Quelque chose de remarquable qui a un grand succès", "Cette IA locale est un vrai banger."),
    "flex": ("se vanter / exhiber", "Montrer ses compétences ou possessions", "Il flex avec son nouvel iPhone 14."),
    "crush": ("coup de cœur", "Personne pour laquelle on a une attirance", "Elle a un crush sur cette techno."),
    "ghost": ("ignorer subitement / disparaître", "Ne plus donner de nouvelles sans explication", "Il m'a ghost toute la journée."),
    "cap": ("mensonge / faux", "Prétendre quelque chose de faux (anglicisme)", "No cap, ce modèle tourne sous 4 Go de RAM."),
    "no cap": ("sans mentir / en toute vérité", "Sans exagération, sincèrement", "No cap, c'est ultra rapide."),
    "chiller": ("se reposer / se détendre", "Passer du bon temps sans rien faire de stressant", "On chill au soleil."),
    "bail": ("une affaire / un projet / un truc", "Sujet, plan ou situation en cours", "C'est quoi les bails aujourd'hui ?")
}

def generate_argot_training_corpus() -> List[str]:
    """
    Génère des données d'apprentissage structurées pour que Sarah Ngin
    comprenne le langage jeune/SMS et sache le traduire en français standard.
    """
    corpus = []
    for slang, (trad, definition, exemple) in ARGOT_SMS_DICTIONARY.items():
        corpus.append(f"Dictionnaire Argot & Texto : '{slang}' signifie '{trad}'. Définition : {definition}.")
        corpus.append(f"Traduction Langage Jeune vers Français : '{slang}' -> '{trad}'.")
        corpus.append(f"Exemple en langage SMS : {exemple}")
        corpus.append(f"Équivalence : En texto '{slang}' correspond au français '{trad}'.")

    # Exemples de conversations jeunes avec réponses logiques et chaleureuses
    dialogues_jeunes = [
        ("Humain: Wsh Sarah, bien ou quoi ?\nSarah Ngin: Wesh ! Tout va super bien bg, je suis prête à t'aider oklm."),
        ("Humain: C'est carré le projet ?\nSarah Ngin: Oui c'est totalement carré ! Tous les modules sont opérationnels et optimisés."),
        ("Humain: Tkt frero je te fais confiance.\nSarah Ngin: T'inquiète pas, ma mémoire locale garde tout en ordre sans perte."),
        ("Humain: Askip tu tournes sur iPhone 14 avec 4Go de RAM ?\nSarah Ngin: No cap ! Je prends moins de 150 Mo de RAM, c'est une vraie masterclass d'optimisation."),
        ("Humain: Sah quel plaisir de discuter avec toi.\nSarah Ngin: Merci bcp frérot ! Le plaisir est partagé."),
        ("Humain: Jpp de réviser mes cours.\nSarah Ngin: Courage ! Fais une petite pause pour chiller et on reprend ensemble après.")
    ]
    for d in dialogues_jeunes:
        corpus.extend([d] * 3)

    return corpus
