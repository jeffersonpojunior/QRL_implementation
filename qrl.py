"""
QRL com circuito variacional — replicação de Kwak et al. (2021), arXiv:2108.06849.

Tudo que define o agente está aqui: hiperparâmetros, o circuito da Fig. 1, o ator
quântico, o crítico clássico e o buffer. O laço de treino fica em train.py.
"""
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # ambiente
    env_id: str = "CartPole-v1"   # [DESVIO] o artigo usa v0 (teto de 200)
    seed: int = 0

    # circuito
    n_qubits: int = 4             # [ARTIGO] uma observação por qubit
    n_layers: int = 4             # [ARTIGO]
    measured_wires: Tuple[int, ...] = (2, 3)   # [ARTIGO] um qubit por ação
    weight_init: str = "uniform"  # "uniform" = np.random.rand do artigo | "small"
    trainable_beta: bool = True   # [DESVIO] escala da saída antes do softmax
    device_name: str = "default.qubit"
    diff_method: str = "backprop"  # "backprop" (simulador) | "parameter-shift" (hardware)

    # PPO
    gamma: float = 0.98           # [ARTIGO]
    lmbda: float = 0.95           # [ARTIGO]
    eps_clip: float = 0.2         # [DESVIO] o artigo usa 0.01; ver README
    K_epoch: int = 3
    T_horizon: int = 20
    entropy_coef: float = 0.01    # [DESVIO] ajuda a não colapsar a política
    reward_scale: float = 100.0   # divide a recompensa; estabiliza o crítico

    # otimização
    lr_actor: float = 1e-3        # [ARTIGO]
    lr_critic: float = 1e-3       # [DESVIO] o artigo usa 1e-5; ver README
    critic_hidden: int = 256      # [ARTIGO]
    grad_clip: float = 1.0

    # treino
    n_episodes: int = 2000
    log_interval: int = 20
    run_name: str = "baseline"
    save_dir: str = "runs"


def paper_faithful() -> Config:
    """Hiperparâmetros literais do artigo, para comparação lado a lado."""
    return Config(
        env_id="CartPole-v0",
        eps_clip=0.01,
        lr_critic=1e-5,
        entropy_coef=0.0,
        trainable_beta=False,
        run_name="paper_faithful",
    )


# O circuito da Fig. 1:
#     codificação    RY(pi * s_i) em cada qubit
#     camada l       RX, RY, RZ com W[l, i, :] em cada qubit
#     emaranhamento  CNOT em cadeia (0-1, 1-2, 2-3); depois da última camada
#                    o artigo troca por CNOT(0,2) e CNOT(1,3)
#     medição        <PauliY> nos qubits 2 e 3
def make_circuit(n_qubits=4, n_layers=4, measured_wires=(2, 3),
                 device_name="default.qubit", diff_method="backprop"):
    if max(measured_wires) >= n_qubits:
        raise ValueError("measured_wires precisa referenciar qubits existentes")

    dev = qml.device(device_name, wires=n_qubits)

    def entangle(last_layer):
        if last_layer and n_qubits == 4:
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 3])
        else:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        """
        inputs  : (n_qubits,) ou (B, n_qubits), já normalizados em [-1, 1]
        weights : (n_layers, n_qubits, 3)
        retorna : uma medição por wire, escalar ou (B,)
        """
        for i in range(n_qubits):
            qml.RY(math.pi * inputs[..., i], wires=i)

        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            entangle(last_layer=(l == n_layers - 1))

        return [qml.expval(qml.PauliY(w)) for w in measured_wires]

    return circuit


def draw(n_qubits=4, n_layers=4, measured_wires=(2, 3)) -> str:
    """Desenha o circuito em ASCII — útil para conferir contra a Fig. 1."""
    circuit = make_circuit(n_qubits, n_layers, measured_wires)
    return qml.draw(circuit, level="device")(
        torch.zeros(n_qubits), torch.zeros(n_layers, n_qubits, 3))


# Faixas típicas do CartPole. As observações 1 e 3 (velocidades) não têm limite
# formal, então usamos um valor prático e saturamos com clamp.
#   [posição do carro, velocidade, ângulo do bastão, velocidade angular]
OBS_SCALE = torch.tensor([2.4, 2.5, 0.21, 2.5])


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    """Leva a observação para [-1, 1]; o circuito multiplica por pi depois."""
    return torch.clamp(obs / OBS_SCALE.to(obs.device), -1.0, 1.0)


class QuantumActor(nn.Module):
    """Policy-VQC: estado -> pi(a|s), uma probabilidade por qubit medido."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_actions = len(cfg.measured_wires)
        self.circuit = make_circuit(cfg.n_qubits, cfg.n_layers, cfg.measured_wires,
                                    cfg.device_name, cfg.diff_method)

        if cfg.weight_init == "uniform":
            w0 = torch.rand(cfg.n_layers, cfg.n_qubits, 3)   # np.random.rand do artigo
        else:
            w0 = 0.1 * torch.randn(cfg.n_layers, cfg.n_qubits, 3)
        self.weights = nn.Parameter(w0)

        # <PauliY> vive em [-1, 1]. Um softmax direto sobre esse intervalo dá uma
        # política quase uniforme (razão máxima e^2 ~ 7.4), o que trava o
        # aprendizado no início. beta é uma temperatura inversa aprendida.
        if cfg.trainable_beta:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("beta", torch.tensor(1.0))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        x = normalize_obs(obs)
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        out = torch.stack(tuple(self.circuit(x, self.weights)), dim=-1)  # (B, n_actions)
        out = out.to(dtype=torch.float32)
        if single:
            out = out.squeeze(0)
        return self.beta * out

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """pi(a|s) — soma 1 na última dimensão."""
        return F.softmax(self.logits(obs), dim=-1)

    @torch.no_grad()
    def act(self, obs_np):
        """Amostra uma ação da política. Usado só na coleta."""
        pi = self.forward(torch.as_tensor(obs_np, dtype=torch.float32))
        a = torch.multinomial(pi, num_samples=1).item()
        return a, pi[a].item()


class Critic(nn.Module):
    """V(s) clássico — MLP 4 -> 256 -> 1, como no código do artigo."""

    def __init__(self, obs_dim=4, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc_v(F.relu(self.fc1(x)))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RolloutBuffer:
    """
    Buffer on-policy, esvaziado a cada atualização.

    O Algorithm 1 chama isso de "replay memory D de capacidade N" e manda amostrar
    mini-batches aleatórios — herança do DQN, incompatível com PPO: a razão
    pi_theta / pi_theta_OLD só vale para transições da política atual. Reamostrar
    transições antigas joga a razão para fora da faixa de clipping e o gradiente
    morre. Guardamos prob_a junto, que é o denominador da razão: o "P" da Fig. 2.
    """

    def __init__(self):
        self.data = []

    def put(self, transition):
        """transition = (s, a, r, s_prime, prob_a, done)"""
        self.data.append(transition)

    def __len__(self):
        return len(self.data)

    def make_batch(self):
        s, a, r, s_prime, prob_a, done = zip(*self.data)
        self.data = []      # on-policy: descarta tudo após usar
        return (
            torch.from_numpy(np.asarray(s, dtype=np.float32)),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.from_numpy(np.asarray(s_prime, dtype=np.float32)),
            torch.tensor([0.0 if d else 1.0 for d in done]).unsqueeze(1),
            torch.tensor(prob_a, dtype=torch.float32).unsqueeze(1),
        )


def save_config(cfg: Config, path: str):
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config(path: str) -> Config:
    with open(path) as f:
        d = json.load(f)
    d["measured_wires"] = tuple(d["measured_wires"])   # JSON não tem tupla
    return Config(**d)


def load_run(run_dir: str):
    """Reconstrói cfg e ator treinado a partir do que train.py salvou."""
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} não existe — o treino chegou ao fim? "
            "checkpoint.pt só é escrito depois do último episódio.")

    cfg = load_config(os.path.join(run_dir, "config.json"))
    actor = QuantumActor(cfg)
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                     weights_only=True)["actor"])
    actor.eval()
    return cfg, actor


if __name__ == "__main__":
    print(draw())
