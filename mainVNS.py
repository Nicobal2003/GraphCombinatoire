import argparse
import time
import numpy as np

from read_input import read_input
from GivenMethod import fobj
from VNStabou import vns_factorization


def write_solution(path: str, fval: float, W: np.ndarray, H: np.ndarray) -> None:
    """Format: f (ligne 1), puis W (m lignes), puis H (r lignes)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{int(round(fval))}\n")
        for i in range(W.shape[0]):
            f.write(" ".join(str(int(x)) for x in W[i]) + "\n")
        for i in range(H.shape[0]):
            f.write(" ".join(str(int(x)) for x in H[i]) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Greedy (bicliques) + VNS (tabou léger) pour X ≈ W H (entier, borné)."
    )
    # I/O
    p.add_argument("--in", dest="inp", default="input.txt", help="Chemin du fichier instance.")
    p.add_argument("--out", dest="out", default="output.txt", help="Chemin du fichier solution.")
    p.add_argument("--quiet", action="store_true", help="Réduit les logs.")
    # VNS knobs
    p.add_argument("--k_max", type=int, default=5, help="Intensité max du shaking.")
    p.add_argument("--max_ls_iters", type=int, default=50, help="Itérations descente locale.")
    p.add_argument("--moves_per_iter", type=int, default=2000, help="Mouvements testés/itération.")
    p.add_argument("--max_no_improv", type=int, default=20, help="Tours VNS sans amélioration.")
    p.add_argument("--time_limit", type=float, default=None, help="Limite dure (s).")
    p.add_argument("--seed", type=int, default=None, help="Graine RNG.")
    # Tabou
    p.add_argument("--tabu_tenure", type=int, default=200, help="Durée tabou (acceptations).")
    # Greedy front-end
    p.add_argument("--use_greedy", type=int, choices=[0, 1], default=1,
                   help="1=activer la construction gloutonne, 0=désactiver.")
    p.add_argument("--greedy_pivots", type=int, default=8,
                   help="Nb de pivots testés par facteur (greedy).")
    p.add_argument("--greedy_min_gain", type=int, default=2,
                   help="Gain minimal pour accepter un facteur greedy.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    X, m, n, r, LW, UW, LH, UH = read_input(args.inp)
    if not args.quiet:
        print(f"[info] m={m} n={n} r={r}  W∈[{LW},{UW}]  H∈[{LH},{UH}]")

    start = time.time()
    W, H, F_algo = vns_factorization(
        X, r, LW, UW, LH, UH,
        k_max=args.k_max,
        max_ls_iters=args.max_ls_iters,
        moves_per_iter=args.moves_per_iter,
        tabu_tenure=args.tabu_tenure,
        max_no_improv=args.max_no_improv,
        time_limit=args.time_limit,
        seed=args.seed,
        use_greedy=bool(args.use_greedy),
        greedy_pivots=args.greedy_pivots,
        greedy_min_gain=args.greedy_min_gain,
    )
    elapsed = time.time() - start

    F_true = fobj(X, W, H)
    if abs(F_true - F_algo) > 1e-6 and not args.quiet:
        print(f"[warn] Algo F={F_algo:.0f} ≠ fobj={F_true:.0f}. On sort avec fobj.")

    write_solution(args.out, F_true, W, H)

    if not args.quiet:
        print(f"[done] f={int(round(F_true))}  time={elapsed:.3f}s  out='{args.out}'")


if __name__ == "__main__":
    main()