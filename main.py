import time
import numpy as np
from fact_in_z_tabu import metaheuristicTabu
from fact_in_z_tabuStochastique import metaheuristicTabuSto
import matplotlib.pyplot as plt

def fobj(X, W, H):
    """
    Retourne ||X - W H||_F^2.
    """
    R = X - W @ H
    return int((R ** 2).sum())

#lire input
def read_instance(path):
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    m, n, r, LW, UW, LH, UH = map(int, lines[0].split())
    X = np.array([[int(x) for x in ln.split()] for ln in lines[1:]])
    return X, m, n, r, LW, UW, LH, UH

#écrire input
def write_solution(path, fval, W, H):
    lines = []
    lines.append(str(fval))
    for i in range(W.shape[0]):
        lines.append(" ".join(str(int(v)) for v in W[i]))
    for i in range(H.shape[0]):
        lines.append(" ".join(str(int(v)) for v in H[i]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    X, m, n, r, LW, UW, LH, UH = read_instance("input.txt")

    # critère de "grosse" matrice, a ajuster si besoin
    num_cells = m * n
    is_big = (num_cells > 10000)   #adapter ici

    print(f"Instance : {m} x {n}, r={r}, W in [{LW},{UW}], H in [{LH},{UH}]")
    print("Mode utilisé :", "Tabu stochastique" if is_big else "Tabu complet (non stochastique)")

    start = time.time()

    if is_big:
        # GROSSE MATRICE > Tabu stochastique (multi start si paramétré)
        W_best, H_best, history = metaheuristicTabuSto(X, r, LW, UW, LH, UH)
    else:
        # PETITE MATRICE > Tabu complet (non stochastique et multi start)
        W_best, H_best, history = metaheuristicTabu(X, r, LW, UW, LH, UH)

    end = time.time()

    f_best = fobj(X, W_best, H_best)
    print("Meilleure valeur trouvée :", f_best)
    print(f"Temps d'exécution : {end - start:.4f} secondes")

    out_name = "output_grosse.txt" if is_big else "output_petite.txt"
    write_solution(out_name, f_best, W_best, H_best)

    # l'historique
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title("Évolution de l erreur f au fil des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("f(X - W H)^2")
    plt.grid(True)
    plt.show()


    # ---------- Visu matrice comme image ----------
    # Reconstruction approchée
    Y = W_best @ H_best

    # Pour garder la même échelle de couleurs que X
    vmin = X.min()
    vmax = X.max()

    plt.figure(figsize=(12, 5))

    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Matrice originale X")
    plt.axis("off")

    # Image reconstruite
    plt.subplot(1, 2, 2)
    plt.imshow(Y, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Reconstruction W @ H")
    plt.axis("off")

    plt.suptitle("Comparaison avant / après (X vs W@H)")
    plt.tight_layout()
    plt.show()

