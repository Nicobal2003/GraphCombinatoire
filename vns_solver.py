import numpy as np
import random
import time
import argparse
from read_input import read_input
from main import write_output, evaluate

# Constants
LARGE_MATRIX_THRESHOLD = 10000
VERY_LARGE_MATRIX_THRESHOLD = 100000
INITIAL_SOLUTION_TRIALS = 5
LOCAL_SEARCH_ITERATIONS_SMALL = 50
LOCAL_SEARCH_ITERATIONS_LARGE = 20
LOCAL_SEARCH_ITERATIONS_VERY_LARGE = 100
SAMPLE_SIZE_LARGE = 2000
SVD_ERROR_THRESHOLD = 0.5
INITIAL_REFINEMENT_ITERATIONS = 10
INITIAL_REFINEMENT_TRIALS = 100

def generate_initial_solution(m, n, r, LW, UW, LH, UH, X=None, use_svd=True):
    """Try SVD first, fallback to random if it fails or is bad"""
    if use_svd and X is not None:
        try:
            U, s, Vt = np.linalg.svd(X.astype(float), full_matrices=False)
            U_r = U[:, :r]
            s_r = s[:r]
            Vt_r = Vt[:r, :]
            
            W_approx = U_r @ np.diag(np.sqrt(s_r))
            H_approx = np.diag(np.sqrt(s_r)) @ Vt_r
            
            # Scale to match X magnitude
            X_mean = np.mean(np.abs(X))
            WH_approx = W_approx @ H_approx
            WH_mean = np.mean(np.abs(WH_approx)) if np.any(WH_approx != 0) else 1
            
            if WH_mean > 0:
                scale_factor = np.sqrt(X_mean / WH_mean) if WH_mean > 0 else 1.0
                W_approx = W_approx * np.sqrt(scale_factor)
                H_approx = H_approx * np.sqrt(scale_factor)
            
            W_abs_max = np.max(np.abs(W_approx)) if np.any(W_approx != 0) else 1
            H_abs_max = np.max(np.abs(H_approx)) if np.any(H_approx != 0) else 1
            
            if W_abs_max > 0:
                W = (W_approx / W_abs_max * max(abs(LW), abs(UW))).astype(int)
            else:
                W = np.zeros((m, r), dtype=int)
            
            if H_abs_max > 0:
                H = (H_approx / H_abs_max * max(abs(LH), abs(UH))).astype(int)
            else:
                H = np.zeros((r, n), dtype=int)
            
            W = np.clip(W, LW, UW)
            H = np.clip(H, LH, UH)
            
            # If SVD gives a bad solution, try random + local refinement
            test_f = evaluate({"W": W, "H": H}, X)
            X_norm_sq = np.sum(X ** 2)
            if test_f > SVD_ERROR_THRESHOLD * X_norm_sq:
                W = np.random.randint(LW, UW + 1, size=(m, r))
                H = np.random.randint(LH, UH + 1, size=(r, n))
                for _ in range(INITIAL_REFINEMENT_ITERATIONS):
                    best_w, best_h = W.copy(), H.copy()
                    best_err = evaluate({"W": W, "H": H}, X)
                    for i in range(min(INITIAL_REFINEMENT_TRIALS, m*r)):
                        idx = np.unravel_index(np.random.randint(0, W.size), W.shape)
                        old_val = W[idx]
                        for delta in [-1, 1]:
                            new_val = np.clip(old_val + delta, LW, UW)
                            W[idx] = new_val
                            err = evaluate({"W": W, "H": H}, X)
                            if err < best_err:
                                best_err = err
                                best_w = W.copy()
                            else:
                                W[idx] = old_val
                    W = best_w
                    if best_err < test_f * 0.9:
                        break
            
            return {"W": W, "H": H}
        except (np.linalg.LinAlgError, ValueError, TypeError):
            pass
    
    W = np.random.randint(LW, UW + 1, size=(m, r))
    H = np.random.randint(LH, UH + 1, size=(r, n))
    return {"W": W, "H": H}

def local_search_1(solution, X, LW, UW, LH, UH, max_iterations=100, sample_size=None):
    """Greedy local search - try changing one element at a time"""
    W, H = solution["W"].copy(), solution["H"].copy()
    current_f = evaluate({"W": W, "H": H}, X)
    improved = True
    iterations = 0
    
    m, n = X.shape
    total_size = m * n
    is_large = total_size > LARGE_MATRIX_THRESHOLD
    
    if is_large and sample_size is None:
        sample_size = min(SAMPLE_SIZE_LARGE, W.size + H.size)
    
    if is_large:
        deltas = [-3, -2, -1, 1, 2, 3]
    else:
        deltas = [-1, 1]
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        if is_large and sample_size is not None:
            num_w_samples = min(sample_size // 2, W.size)
            num_h_samples = min(sample_size // 2, H.size)
            w_indices = [(np.random.randint(0, W.shape[0]), 
                         np.random.randint(0, W.shape[1])) 
                        for _ in range(num_w_samples)]
            h_indices = [(np.random.randint(0, H.shape[0]), 
                         np.random.randint(0, H.shape[1])) 
                        for _ in range(num_h_samples)]
        else:
            w_indices = [(i, j) for i in range(W.shape[0]) for j in range(W.shape[1])]
            h_indices = [(i, j) for i in range(H.shape[0]) for j in range(H.shape[1])]
        
        for i, j in w_indices:
            for delta in deltas:
                new_value = W[i, j] + delta
                if LW <= new_value <= UW:
                    old_value = W[i, j]
                    W[i, j] = new_value
                    new_f = evaluate({"W": W, "H": H}, X)
                    if new_f < current_f:
                        current_f = new_f
                        improved = True
                        break
                    else:
                        W[i, j] = old_value
            if improved:
                break
        
        if not improved:
            for i, j in h_indices:
                for delta in deltas:
                    new_value = H[i, j] + delta
                    if LH <= new_value <= UH:
                        old_value = H[i, j]
                        H[i, j] = new_value
                        new_f = evaluate({"W": W, "H": H}, X)
                        if new_f < current_f:
                            current_f = new_f
                            improved = True
                            break
                        else:
                            H[i, j] = old_value
                if improved:
                    break
    
    return {"W": W, "H": H}, current_f

def shake_neighborhood_2(solution, LW, UW, LH, UH, intensity=1):
    """Change a whole row or column"""
    W, H = solution["W"].copy(), solution["H"].copy()
    
    if random.random() < 0.5:
        row_idx = random.randint(0, W.shape[0] - 1)
        for j in range(W.shape[1]):
            delta = random.randint(-intensity, intensity)
            new_value = W[row_idx, j] + delta
            W[row_idx, j] = np.clip(new_value, LW, UW)
    else:
        col_idx = random.randint(0, H.shape[1] - 1)
        for i in range(H.shape[0]):
            delta = random.randint(-intensity, intensity)
            new_value = H[i, col_idx] + delta
            H[i, col_idx] = np.clip(new_value, LH, UH)
    
    return {"W": W, "H": H}

def shake_neighborhood_3(solution, LW, UW, LH, UH):
    """Swap two rows or two columns"""
    W, H = solution["W"].copy(), solution["H"].copy()
    
    if random.random() < 0.5 and W.shape[0] > 1:
        i1, i2 = random.sample(range(W.shape[0]), 2)
        W[[i1, i2]] = W[[i2, i1]]
    elif H.shape[1] > 1:
        j1, j2 = random.sample(range(H.shape[1]), 2)
        H[:, [j1, j2]] = H[:, [j2, j1]]
    
    return {"W": W, "H": H}

def shake_neighborhood_4(solution, LW, UW, LH, UH, num_changes=5):
    """Randomly tweak a few elements"""
    W, H = solution["W"].copy(), solution["H"].copy()
    
    for _ in range(num_changes):
        i = random.randint(0, W.shape[0] - 1)
        j = random.randint(0, W.shape[1] - 1)
        delta = random.randint(-2, 2)
        new_value = W[i, j] + delta
        W[i, j] = np.clip(new_value, LW, UW)
    
    for _ in range(num_changes):
        i = random.randint(0, H.shape[0] - 1)
        j = random.randint(0, H.shape[1] - 1)
        delta = random.randint(-2, 2)
        new_value = H[i, j] + delta
        H[i, j] = np.clip(new_value, LH, UH)
    
    return {"W": W, "H": H}

def shake_neighborhood_5(solution, LW, UW, LH, UH, fraction=0.1):
    """Reset part of the solution randomly"""
    W, H = solution["W"].copy(), solution["H"].copy()
    
    num_changes = int(W.size * fraction)
    indices = random.sample(range(W.size), num_changes)
    for idx in indices:
        i, j = np.unravel_index(idx, W.shape)
        W[i, j] = random.randint(LW, UW)
    
    num_changes = int(H.size * fraction)
    indices = random.sample(range(H.size), num_changes)
    for idx in indices:
        i, j = np.unravel_index(idx, H.shape)
        H[i, j] = random.randint(LH, UH)
    
    return {"W": W, "H": H}

def variable_neighborhood_search(X, m, n, r, LW, UW, LH, UH,
                                max_iterations=1000,
                                max_no_improvement=50,
                                k_max=5,
                                time_limit=None):
    """Main VNS loop - shake, local search, repeat
    
    Args:
        time_limit: Maximum time in seconds (None for no limit)
    """
    search_start_time = time.time()
    
    print(f"\nInitialisation:")
    print(f"   Dimensions: {m}×{n}, rang {r}")
    print(f"   Bornes W: [{LW}, {UW}], Bornes H: [{LH}, {UH}]")
    print(f"   Elements: {m*r + r*n}")
    if time_limit is not None:
        print(f"   Limite de temps: {time_limit} secondes")
    
    print(f"   Generation de solutions initiales...")
    best_init_solution = None
    best_init_f = float('inf')
    
    svd_solution = generate_initial_solution(m, n, r, LW, UW, LH, UH, X, use_svd=True)
    svd_f = evaluate(svd_solution, X)
    if svd_f < best_init_f:
        best_init_solution = svd_solution
        best_init_f = svd_f
        print(f"   Solution SVD: erreur = {svd_f:.2f}")
    
    for i in range(INITIAL_SOLUTION_TRIALS):
        random_solution = generate_initial_solution(m, n, r, LW, UW, LH, UH, X, use_svd=False)
        random_f = evaluate(random_solution, X)
        if random_f < best_init_f:
            best_init_solution = random_solution
            best_init_f = random_f
    
    current_solution = best_init_solution
    current_f = best_init_f
    
    best_solution = {"W": current_solution["W"].copy(), 
                     "H": current_solution["H"].copy()}
    best_f = current_f
    
    no_improvement_count = 0
    iteration = 0
    total_improvements = 0
    neighborhood_stats = {i: 0 for i in range(1, k_max + 1)}
    
    total_size = m * n
    is_large = total_size > LARGE_MATRIX_THRESHOLD
    
    if is_large:
        local_max_iter = LOCAL_SEARCH_ITERATIONS_LARGE
        if total_size > VERY_LARGE_MATRIX_THRESHOLD:
            max_iterations = min(max_iterations, 200)
            max_no_improvement = min(max_no_improvement, 30)
        print(f"   Mode: Grande matrice (echantillonnage active)")
    else:
        local_max_iter = LOCAL_SEARCH_ITERATIONS_SMALL
        print(f"   Mode: Petite matrice (recherche exhaustive)")
    
    print(f"\nSolution initiale: erreur = {best_f:.2f}")
    print(f"Parametres: max_iterations={max_iterations}, max_no_improvement={max_no_improvement}")
    print(f"\n{'='*60}")
    print(f"Debut de la recherche VNS...")
    print(f"{'='*60}\n")
    
    while iteration < max_iterations and no_improvement_count < max_no_improvement:
        # Check time limit
        if time_limit is not None:
            elapsed_time = time.time() - search_start_time
            if elapsed_time >= time_limit:
                print(f"\nLimite de temps atteinte ({time_limit}s)")
                break
        iteration += 1
        k = 1
        improvement_in_iteration = False
        
        while k <= k_max:
            if k == 1:
                shaken_solution = shake_neighborhood_2(current_solution, LW, UW, LH, UH, intensity=1)
            elif k == 2:
                shaken_solution = shake_neighborhood_2(current_solution, LW, UW, LH, UH, intensity=2)
            elif k == 3:
                shaken_solution = shake_neighborhood_3(current_solution, LW, UW, LH, UH)
            elif k == 4:
                shaken_solution = shake_neighborhood_4(current_solution, LW, UW, LH, UH, num_changes=5)
            else:
                shaken_solution = shake_neighborhood_5(current_solution, LW, UW, LH, UH, fraction=0.1)
            
            neighborhood_stats[k] += 1
            
            if is_large:
                local_max_iter_actual = min(LOCAL_SEARCH_ITERATIONS_VERY_LARGE, local_max_iter * 2)
            else:
                local_max_iter_actual = local_max_iter
            
            local_solution, local_f = local_search_1(shaken_solution, X, LW, UW, LH, UH, 
                                                    max_iterations=local_max_iter_actual)
            
            if local_f < current_f:
                improvement_amount = current_f - local_f
                current_solution = {"W": local_solution["W"].copy(), 
                                   "H": local_solution["H"].copy()}
                current_f = local_f
                k = 1
                improvement_in_iteration = True
                
                if local_f < best_f:
                    best_solution = {"W": local_solution["W"].copy(), 
                                    "H": local_solution["H"].copy()}
                    improvement_amount_best = best_f - local_f
                    best_f = local_f
                    no_improvement_count = 0
                    total_improvements += 1
                    
                    print(f"Iteration {iteration:4d} | Voisinage {k} | "
                          f"Amelioration: {improvement_amount_best:.2f} | "
                          f"Nouvelle meilleure: {best_f:.2f} | "
                          f"Pas d'amelioration: {no_improvement_count}/{max_no_improvement}")
                else:
                    print(f"  -> Iteration {iteration:4d} | Voisinage {k} | "
                          f"Amelioration locale: {improvement_amount:.2f} | "
                          f"Erreur actuelle: {current_f:.2f}")
            else:
                k += 1
        
        if not improvement_in_iteration:
            no_improvement_count += 1
            if iteration % 10 == 0 or no_improvement_count % 10 == 0:
                elapsed = time.time() - search_start_time
                time_str = f" | Temps: {elapsed:.1f}s" if time_limit is not None else ""
                print(f"Iteration {iteration:4d} | Pas d'amelioration | "
                      f"Meilleure: {best_f:.2f} | "
                      f"Compteur: {no_improvement_count}/{max_no_improvement}{time_str}")
    
    total_time = time.time() - search_start_time
    print(f"\n{'='*60}")
    print(f"Recherche terminee")
    print(f"{'='*60}")
    print(f"\nStatistiques:")
    print(f"   Iterations totales: {iteration}")
    print(f"   Ameliorations trouvees: {total_improvements}")
    print(f"   Iterations sans amelioration: {no_improvement_count}")
    print(f"   Temps de recherche: {total_time:.2f} secondes")
    if time_limit is not None:
        print(f"   Limite de temps: {time_limit}s ({'atteinte' if total_time >= time_limit else 'non atteinte'})")
    print(f"   Utilisation des voisinages:")
    for k, count in neighborhood_stats.items():
        percentage = (count / sum(neighborhood_stats.values())) * 100 if sum(neighborhood_stats.values()) > 0 else 0
        print(f"     * Voisinage {k}: {count} fois ({percentage:.1f}%)")
    
    return best_solution, best_f

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Variable Neighborhood Search for matrix factorization')
    parser.add_argument('--input', type=str, default='input.txt', help='Input file path')
    parser.add_argument('--output', type=str, default='output_vns.txt', help='Output file path')
    parser.add_argument('--time-limit', type=float, default=None, 
                        help='Maximum time in seconds (e.g., 30 for 30 seconds)')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    time_limit = args.time_limit
    
    X, m, n, r, LW, UW, LH, UH = read_input(input_path)
    
    print("="*60)
    print("--- Lancement du VNS ---")
    print("="*60)
    
    start_time = time.time()
    
    total_size = m * n
    if total_size > VERY_LARGE_MATRIX_THRESHOLD:
        max_iter = 1000
        max_no_imp = 100
        print(f"\nMatrice grande detectee ({m}×{n})")
        print(f"   Utilisation de parametres optimises")
    elif total_size > LARGE_MATRIX_THRESHOLD:
        max_iter = 1500
        max_no_imp = 150
        print(f"\nMatrice moyenne detectee ({m}×{n})")
        print(f"   Utilisation de parametres adaptes")
    else:
        max_iter = 500
        max_no_imp = 100
        print(f"\nMatrice petite detectee ({m}×{n})")
        print(f"   Utilisation de parametres standards")
    
    best_solution, best_f = variable_neighborhood_search(
        X, m, n, r, LW, UW, LH, UH,
        max_iterations=max_iter,
        max_no_improvement=max_no_imp,
        k_max=5,
        time_limit=time_limit
    )
    
    execution_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("--- Resultat final ---")
    print("="*60)
    print("\nMatrice W:")
    print(best_solution["W"])
    print("\nMatrice H:")
    print(best_solution["H"])
    print(f"\nValeur finale de la fonction objectif = {best_f}")
    print(f"Temps execution VNS = {execution_time:.6f} secondes")
    
    if execution_time > 60:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"   ({minutes} minute(s) et {seconds:.2f} seconde(s))")
    
    write_output(output_path, best_f, best_solution["W"], best_solution["H"])
    print(f"\nResultats sauvegardes dans: {output_path}")
    print("="*60)

