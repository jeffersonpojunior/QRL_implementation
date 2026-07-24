"""
Gera a figura equivalente à Fig. 4 do artigo a partir dos CSVs de treino.

    python plot_results.py runs/baseline_seed0 runs/baseline_seed1 --label "quântico"
    python plot_results.py runs/*_seed* --out fig4_replicacao.png
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scores(run_dir: str):
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return np.array([float(row["score"]) for row in csv.DictReader(f)])


def smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="diretórios de run (aceita glob)")
    p.add_argument("--label", default="Agente quântico")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--random-baseline", type=float, default=None,
                   help="linha horizontal da política aleatória (~22 no CartPole)")
    p.add_argument("--out", default="resultados.png")
    args = p.parse_args()

    dirs = []
    for r in args.runs:
        dirs.extend(sorted(glob.glob(r)) if "*" in r else [r])

    curves = [smooth(load_scores(d), args.window) for d in dirs]
    n = min(len(c) for c in curves)
    arr = np.stack([c[:n] for c in curves])
    mean, std = arr.mean(0), arr.std(0)
    x = np.arange(n) + args.window

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, mean, lw=2, color="#00a48f", label=f"{args.label} (n={len(dirs)})")
    if len(dirs) > 1:
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="#00a48f")
    if args.random_baseline:
        plt.axhline(args.random_baseline, ls="--", lw=1.5, color="#9d95bb",
                    label="Política aleatória")

    plt.xlabel("Episódio")
    plt.ylabel(f"Recompensa total (média móvel, janela {args.window})")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        plt.gca().spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)

    print(f"figura salva em {args.out}")
    print(f"runs: {len(dirs)} | episódios: {n + args.window}")
    print(f"score final (média das últimas 100): {arr[:, -100:].mean():.1f}")


if __name__ == "__main__":
    main()
