"""
Testa o agente treinado.

A curva do log.csv é o desempenho DURANTE o aprendizado: política ainda mudando,
exploração ligada. Aqui a política fica congelada e é medida em episódios novos,
com seeds que não apareceram no treino. O crítico não participa — V(s) só existe
para calcular a vantagem.

    python evaluate.py runs/baseline_seed0
    python evaluate.py runs/baseline_seed0 --greedy --episodes 100 --random
    python evaluate.py runs/baseline_seed0 --render
"""
import argparse
import os
import statistics

import gymnasium as gym
import numpy as np
import torch

from qrl import load_run, n_params


@torch.no_grad()
def run_episodes(env, n_episodes, seed0, actor=None, greedy=False):
    """actor=None roda a política aleatória, para servir de baseline."""
    env.action_space.seed(seed0)     # a baseline aleatória também precisa ser reprodutível
    scores = []
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed0 + i)
        done, score = False, 0.0

        while not done:
            if actor is None:
                a = env.action_space.sample()
            else:
                pi = actor(torch.as_tensor(s, dtype=torch.float32))
                # greedy mostra o que a política de fato aprendeu; amostrar
                # reproduz o comportamento do treino.
                a = int(pi.argmax()) if greedy else int(torch.multinomial(pi, 1))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            score += r

        scores.append(score)
    return scores


def report(name, scores):
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"\n{name}")
    print(f"  média     : {mean:.1f} ± {std:.1f}  ({len(scores)} episódios)")
    print(f"  mediana   : {statistics.median(scores):.1f}")
    print(f"  min / max : {min(scores):.0f} / {max(scores):.0f}")
    # O achado central do artigo é a variância; o CV resume isso num número
    # comparável entre runs de escala diferente.
    if mean > 0:
        print(f"  CV        : {std / mean:.2f}")
    return mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="diretório do run, ex.: runs/baseline_seed0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--greedy", action="store_true",
                   help="argmax da política em vez de amostrar")
    p.add_argument("--seed", type=int, default=10_000,
                   help="seed inicial dos episódios de teste; alta de propósito, "
                        "para não colidir com as do treino")
    p.add_argument("--render", action="store_true", help="abre a janela do env")
    p.add_argument("--random", action="store_true",
                   help="mede também a política aleatória, para comparação")
    args = p.parse_args()

    cfg, actor = load_run(args.run_dir)
    print(f"run      : {args.run_dir}  ({cfg.env_id}, {n_params(actor)} parâmetros)")
    print(f"política : {'greedy (argmax)' if args.greedy else 'amostrada'}")

    env = gym.make(cfg.env_id, render_mode="human" if args.render else None)
    scores = run_episodes(env, args.episodes, args.seed, actor, args.greedy)
    env.close()
    mean = report("AGENTE QUÂNTICO", scores)

    if args.random:
        env = gym.make(cfg.env_id)
        baseline = run_episodes(env, args.episodes, args.seed)
        env.close()
        mean_random = report("POLÍTICA ALEATÓRIA (mesmas seeds)", baseline)
        print(f"\nganho sobre a aleatória: {mean / max(mean_random, 1e-9):.2f}×")

    np.save(os.path.join(args.run_dir, "eval_scores.npy"), np.array(scores))


if __name__ == "__main__":
    main()
