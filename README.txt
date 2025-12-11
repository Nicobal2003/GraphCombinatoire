===============================
README.txt — Paramètres & Usage
===============================

Ce programme résout le problème FactInZ en utilisant une Recherche Tabou Stochastique optimisée pour les grosses matrices (images 750x550, 1000x1000, etc.).



FORMAT DU FICHIER D'ENTRÉE

Le fichier input.txt doit avoir cette structure :

m n r LW UW LH UH
<m lignes contenant n entiers représentant la matrice X>

Exemple :

750 550 10 -16 16 -16 16
123 45 67 ... (550 valeurs)
... (750 lignes)

m = nombre de lignes

n = nombre de colonnes

r = rang recherché

LW, UW = bornes pour W

LH, UH = bornes pour H




FORMAT DU FICHIER DE SORTIE

Le programme crée output_grosse.txt contenant :

La meilleure valeur f trouvée

La matrice W (m lignes de r entiers)

La matrice H (r lignes de n entiers)




PARAMÈTRES MODIFIABLES

Les paramètres importants sont dans :

tabuSto_search_fact_in_z(...)

a) max_iter

nombre maximal d'itérations

plus grand = meilleure solution, mais plus lent
Valeur classique : 8000 à 20000

b) tabu_tenure

durée pendant laquelle inverse-mouvement est tabou

10 = exploration agressive

20–40 = diversification plus large
Valeur typique : 20

c) max_no_improve

arrêt si plus d’amélioration pendant X itérations
Valeur typique : 1000

d) n_candidates_W / n_candidates_H

nombre d’essais aléatoires dans le voisinage

paramètre le plus important pour la qualité
200 = rapide
800 = bon compromis
1200+ = très lent mais plus performant
Valeur typique : 800

e) use_svd_init
True = meilleure initialisation (recommandé pour images)
False = random pur (utile pour multi-start)



MULTI-START

La fonction :

metaheuristicTabuSto(..., n_restarts)

permet d’exécuter plusieurs Tabu successifs avec des initialisations différentes.

n_restarts = 1 → rapide
n_restarts = 3–5 → meilleure qualité
n_restarts = 10+ → coûteux mais robuste



AFFICHAGE ET TIMER

Le programme affiche :

[Tabu] it=XXXX, f=YYYYY, best=ZZZZ, temps = TT.s

Cela montre :

l’itération courante

la valeur actuelle de f

la meilleure valeur rencontrée

le temps écoulé en secondes




CONSEILS DE RÉGLAGE

Pour meilleure qualité :

augmenter max_iter

augmenter n_candidates_W/H

augmenter max_no_improve

garder use_svd_init = True


Pour aller plus vite :

diminuer n_candidates_W/H

diminuer max_iter




REMARQUE IMPORTANTE

FactInZ est un problème difficile.
L’algorithme produit une bonne approximation, pas un optimum garanti.
Les résultats dépendent fortement :

du rang r

des bornes

de l’initialisation

du voisinage aléatoire