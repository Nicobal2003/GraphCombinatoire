"""
Main qui choisit automatiquement la métaheuristique appropriée selon la taille de la matrice.
Structure identique aux fichiers fact_in : utilise read_instance et write_solution.
"""
import sys
import time
import os

# Import des fonctions communes (même structure que fact_in)
from fact_in_z_tabu import read_instance, write_solution, fobj
from fact_in_z_tabu import metaheuristic as metaheuristic_tabu
from fact_in_z import hybrid_ga_tabu
from fact_in_z_tabuStochastique import metaheuristic as metaheuristic_tabu_stoch


def choose_metaheuristic(m, n):
    """
    Choisit automatiquement la métaheuristique appropriée selon la taille de la matrice.
    - Petites et moyennes matrices : AG + Tabu (meilleure qualité)
    - Grandes matrices : Tabu Stochastique (optimisé pour grandes instances)
    
    Args:
        m: nombre de lignes
        n: nombre de colonnes
    
    Returns:
        str: nom de la métaheuristique à utiliser ("hybrid" ou "tabu_stoch")
    """
    matrix_size = m * n
    
    if matrix_size <= 10000:
        # Petites et moyennes matrices (≤ 10000 éléments) : AG + Tabu (meilleure qualité)
        return "hybrid"
    else:
        # Grandes matrices (> 10000) : Tabu stochastique (optimisé pour grandes instances)
        return "tabu_stoch"


def run_metaheuristic(meta_name, X, m, n, r, LW, UW, LH, UH):
    """
    Exécute la métaheuristique spécifiée et retourne (W, H, f_value).
    Respecte la structure des fichiers fact_in.
    
    Args:
        meta_name: nom de la métaheuristique ("hybrid" ou "tabu_stoch")
        X: matrice d'entrée
        m, n, r: dimensions
        LW, UW, LH, UH: bornes
    
    Returns:
        tuple: (W, H, f_value) - format identique à fact_in
    """
    print(f"\n{'='*60}")
    print(f"=== Métaheuristique choisie: {meta_name.upper()} ===")
    print(f"{'='*60}")
    
    if meta_name == "tabu":
        # fact_in_z_tabu.py : metaheuristic retourne (W, H)
        W, H = metaheuristic_tabu(X, r, LW, UW, LH, UH)
        f_value = fobj(X, W, H)
        return W, H, f_value
    
    elif meta_name == "hybrid":
        # fact_in_z.py : hybrid_ga_tabu retourne (W, H, f_best)
        W, H, f_value = hybrid_ga_tabu(X, r, LW, UW, LH, UH, verbose=True)
        return W, H, f_value
    
    elif meta_name == "tabu_stoch":
        # fact_in_z_tabuStochastique.py : metaheuristic retourne (W, H, history)
        big_instance = (m * n > 10000)
        result = metaheuristic_tabu_stoch(X, r, LW, UW, LH, UH, big_instance=big_instance)
        if len(result) == 3:
            W, H, history = result
        else:
            W, H = result[0], result[1]
        f_value = fobj(X, W, H)
        return W, H, f_value
    
    else:
        raise ValueError(f"Métaheuristique inconnue: {meta_name}")


def main():
    """
    Fonction principale : lit une instance et choisit automatiquement la métaheuristique.
    Structure identique aux fichiers fact_in : utilise read_instance et write_solution.
    """
    # Récupération du fichier d'instance (argument obligatoire)
    if len(sys.argv) < 2:
        print("Usage: python main.py <fichier_matrice>")
        print("Exemple: python main.py testmatrice1_8x6_r4_03m22.txt")
        sys.exit(1)
    
    instance_file = sys.argv[1]
    
    # Vérification de l'existence du fichier
    if not os.path.exists(instance_file):
        print(f"Erreur : le fichier '{instance_file}' n'existe pas.")
        sys.exit(1)
    
    # Lecture de l'instance (même format que fact_in)
    try:
        X, m, n, r, LW, UW, LH, UH = read_instance(instance_file)
        print(f"\n{'='*60}")
        print(f"=== Instance : {instance_file} ===")
        print(f"{'='*60}")
        print(f"Dimensions : {m} x {n} (taille = {m * n} éléments)")
        print(f"Rang : r = {r}")
        print(f"Bornes : W in [{LW}, {UW}], H in [{LH}, {UH}]")
    except FileNotFoundError:
        print(f"Erreur : fichier '{instance_file}' introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Choix automatique de la métaheuristique selon la taille
    meta_name = choose_metaheuristic(m, n)
    print(f"\nTaille de la matrice : {m * n} éléments")
    print(f"→ Métaheuristique choisie automatiquement : {meta_name}")
    
    # Exécution de la métaheuristique
    start_time = time.time()
    try:
        W_best, H_best, f_best = run_metaheuristic(meta_name, X, m, n, r, LW, UW, LH, UH)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Affichage des résultats (même format que fact_in)
        print(f"\n{'='*60}")
        print(f"=== RÉSULTATS FINAUX ===")
        print(f"{'='*60}")
        print(f"Meilleure valeur trouvée : {f_best}")
        print(f"Temps d'exécution : {elapsed_time:.4f} secondes ({elapsed_time/60:.2f} minutes)")
        
        # Écriture de la solution (même format que fact_in)
        base_name, ext = os.path.splitext(instance_file)
        output_file = f"{base_name}_out{ext}"
        write_solution(output_file, f_best, W_best, H_best)
        print(f"\nSolution sauvegardée dans : {output_file}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

