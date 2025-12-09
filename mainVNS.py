# file: mainVNS.py
import argparse
import time
import numpy as np

from read_input import read_input
from GivenMethod import fobj
# IMPORTANT: utiliser la version VNS avec tabou léger
from VNStabou import vns_factorization


def write_solution(path: str, fval: float, W: np.ndarray, H: np.ndarray) -> None:
    """Format attendu: f sur la 1re ligne, puis W (m lignes), puis H (r lignes)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{int(round(fval))}\n")
        for i in range(W.shape[0]):
            f.write(" ".join(str(int(x)) for x in W[i]) + "\n")
        for i in range(H.shape[0]):
            f.write(" ".join(str(int(x)) for x in H[i]) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VNS (+tabou léger) pour X ≈ W H (entier, borné).")
    p.add_argument("--in", dest="inp", default="input.txt", help="Chemin du fichier instance.")
    p.add_argument("--out", dest="out", default="output.txt", help="Chemin du fichier solution.")
    # VNS knobs
    p.add_argument("--k_max", type=int, default=5)
    p.add_argument("--max_ls_iters", type=int, default=50)
    p.add_argument("--moves_per_iter", type=int, default=2000)
    p.add_argument("--max_no_improv", type=int, default=20)
    p.add_argument("--time_limit", type=float, default=None, help="Limite dure en secondes.")
    p.add_argument("--seed", type=int, default=None)
    # Tabou knobs
    p.add_argument("--tabu_tenure", type=int, default=200, help="Durée tabou (acceptations).")
    p.add_argument(
        "--no_reset_tabu_on_shake",
        action="store_true",
        help="Ne pas réinitialiser la mémoire tabou après shaking (par défaut: reset).",
    )
    p.add_argument("--quiet", action="store_true")
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
        reset_tabu_on_shake=(not args.no_reset_tabu_on_shake),
        max_no_improv=args.max_no_improv,
        time_limit=args.time_limit,
        seed=args.seed,
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
