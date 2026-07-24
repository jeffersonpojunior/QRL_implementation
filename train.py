"""
Treino PPO do ator quântico — o Algorithm 1 do artigo.

O laço abaixo segue a estrutura do pseudocódigo: coletar T_horizon transições
(# 1. Inference Process #), depois K_epoch passos de gradiente sobre elas
(# 2. Training Process #). Duas escolhas divergem do papel e estão explicadas no
README: amostramos de pi em vez de epsilon-greedy (a razão r_j do PPO exige
política estocástica) e o buffer é on-policy.
"""
import argparse
import csv
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from qrl import (Config, Critic, QuantumActor, RolloutBuffer, n_params,
                 paper_faithful, save_config)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(delta, gamma, lmbda):
    """A_j = delta_j + (gamma*lambda) delta_{j+1} + (gamma*lambda)^2 delta_{j+2} + ..."""
    advantages, running = [], 0.0
    for d in reversed(delta.squeeze(-1).tolist()):
        running = gamma * lmbda * running + d
        advantages.append(running)
    advantages.reverse()
    return torch.tensor(advantages).unsqueeze(1)


def ppo_update(actor, critic, opt_actor, opt_critic, batch, cfg):
    s, a, r, s_prime, done_mask, prob_a = batch
    stats = {}

    for _ in range(cfg.K_epoch):
        with torch.no_grad():
            td_target = r + cfg.gamma * critic(s_prime) * done_mask

        v_s = critic(s)
        delta = (td_target - v_s).detach()

        advantage = compute_gae(delta, cfg.gamma, cfg.lmbda)
        # Normalizar estabiliza o PPO, mas std() de UMA amostra devolve NaN — e um
        # único NaN contamina theta de forma irreversível.
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8)

        # A razão r_j reexecuta o CIRCUITO com o theta atual: prob_a é pi_theta_OLD
        # e está fora do grafo de autograd, não dá para reaproveitar.
        pi = actor(s)
        pi_a = pi.gather(1, a)
        ratio = torch.exp(torch.log(pi_a + 1e-10) - torch.log(prob_a + 1e-10))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip) * advantage
        entropy = -(pi * torch.log(pi + 1e-10)).sum(-1).mean()

        loss_actor = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy
        loss_critic = F.smooth_l1_loss(v_s, td_target)
        loss = loss_actor + loss_critic

        if not torch.isfinite(loss):
            print("  [aviso] perda não finita — passo descartado")
            continue

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
        opt_actor.step()
        opt_critic.step()

        stats = {"loss_actor": loss_actor.item(), "loss_critic": loss_critic.item(),
                 "entropy": entropy.item()}
    return stats


def train(cfg: Config):
    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)

    actor = QuantumActor(cfg)
    critic = Critic(env.observation_space.shape[0], cfg.critic_hidden)
    opt_actor = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    buf = RolloutBuffer()

    out_dir = os.path.join(cfg.save_dir, f"{cfg.run_name}_seed{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.json"))

    print(f"ator quântico : {n_params(actor)} parâmetros")
    print(f"crítico       : {n_params(critic)} parâmetros")
    print(f"saída em      : {out_dir}\n")

    csv_file = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    log = csv.writer(csv_file)
    log.writerow(["episode", "score", "entropy", "loss_actor", "loss_critic", "elapsed_s"])

    scores, t0 = [], time.time()

    for ep in range(1, cfg.n_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        done, score, stats = False, 0.0, {}

        while not done:
            for _ in range(cfg.T_horizon):        # 1. Inference Process
                a, prob_a = actor.act(s)
                s_prime, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                buf.put((s, a, r / cfg.reward_scale, s_prime, prob_a, done))
                s = s_prime
                score += r
                if done:
                    break

            if len(buf) > 0:                      # 2. Training Process
                stats = ppo_update(actor, critic, opt_actor, opt_critic,
                                   buf.make_batch(), cfg)

        scores.append(score)
        log.writerow([ep, score, stats.get("entropy", 0.0),
                      stats.get("loss_actor", 0.0), stats.get("loss_critic", 0.0),
                      round(time.time() - t0, 1)])

        if ep % cfg.log_interval == 0:
            csv_file.flush()
            avg = sum(scores[-cfg.log_interval:]) / cfg.log_interval
            print(f"ep {ep:5d} | score médio {avg:7.1f} | "
                  f"entropia {stats.get('entropy', 0):.3f} | {time.time() - t0:6.1f}s")

        # Checkpoint periódico: uma run de 1500 episódios leva horas e não vale
        # perder os pesos se ela for interrompida.
        if ep % 100 == 0 or ep == cfg.n_episodes:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                       os.path.join(out_dir, "checkpoint.pt"))

    csv_file.close()
    env.close()
    print(f"\nfinalizado. resultados em {out_dir}")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true",
                   help="usa os hiperparâmetros literais do artigo")
    p.add_argument("--episodes", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-name")
    args = p.parse_args()

    cfg = paper_faithful() if args.paper else Config()
    if args.episodes is not None:
        cfg.n_episodes = args.episodes
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name is not None:
        cfg.run_name = args.run_name

    train(cfg)


if __name__ == "__main__":
    main()
