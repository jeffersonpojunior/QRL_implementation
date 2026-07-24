"""
Testes de sanidade — rode antes de gastar horas num treino.

    python test_circuit.py
"""
import time

import torch

from qrl import (Config, Critic, QuantumActor, draw, make_circuit, n_params,
                 normalize_obs)


def check(name, cond, detail=""):
    print(f"[{'ok ' if cond else 'FALHA'}] {name}" + (f"  — {detail}" if detail else ""))
    assert cond, name


def test_circuit_shapes():
    circuit = make_circuit()
    w = torch.rand(4, 4, 3)
    check("saída de 1 amostra tem 2 medições", len(circuit(torch.zeros(4), w)) == 2)

    batched = circuit(torch.zeros(7, 4), w)
    check("broadcasting funciona", tuple(batched[0].shape) == (7,),
          f"shape {tuple(batched[0].shape)}")


def test_batch_equals_loop():
    """O lote precisa dar o MESMO resultado que executar uma amostra por vez."""
    circuit = make_circuit()
    w = torch.rand(4, 4, 3)
    x = torch.rand(5, 4) * 2 - 1

    batched = torch.stack(tuple(circuit(x, w)), dim=-1)
    looped = torch.stack([torch.stack(tuple(circuit(x[i], w))) for i in range(5)])

    err = (batched - looped).abs().max().item()
    check("lote == laço", err < 1e-5, f"erro máx {err:.2e}")


def test_expval_range():
    circuit = make_circuit()
    w = torch.rand(4, 4, 3) * 6.28
    out = torch.stack(tuple(circuit(torch.rand(64, 4) * 2 - 1, w)), dim=-1)
    check("<PauliY> dentro de [-1, 1]", out.abs().max().item() <= 1.0 + 1e-6,
          f"máx |.| = {out.abs().max().item():.4f}")


def test_policy_is_distribution():
    pi = QuantumActor(Config())(torch.rand(11, 4) * 4 - 2)
    check("pi tem shape (B, 2)", tuple(pi.shape) == (11, 2))
    check("pi soma 1", torch.allclose(pi.sum(-1), torch.ones(11), atol=1e-5))
    check("pi é positiva", (pi > 0).all().item())


def test_normalization():
    x = normalize_obs(torch.tensor([[10.0, -99.0, 0.105, 0.0]]))
    check("normalização satura em [-1, 1]", x.abs().max().item() <= 1.0)
    check("ângulo mapeia proporcionalmente", abs(x[0, 2].item() - 0.5) < 0.01,
          f"{x[0, 2].item():.3f}")


def test_param_counts():
    cfg = Config()
    actor, critic = QuantumActor(cfg), Critic(hidden=cfg.critic_hidden)
    check("ator tem 48 ângulos", actor.weights.numel() == 48)
    check("crítico tem ~1.5k parâmetros", 1200 < n_params(critic) < 1800,
          f"{n_params(critic)}")


def test_gradients_flow_through_circuit():
    """O teste mais importante: o gradiente atravessa a simulação quântica?"""
    cfg = Config()
    actor = QuantumActor(cfg)
    loss = -torch.log(actor(torch.rand(8, 4))[:, 0] + 1e-10).mean()
    loss.backward()

    g = actor.weights.grad
    check("weights.grad existe", g is not None)
    check("gradiente não é nulo", g.abs().sum().item() > 1e-9,
          f"|grad| = {g.abs().sum().item():.3e}")
    check("gradiente sem NaN", torch.isfinite(g).all().item())
    if cfg.trainable_beta:
        check("beta recebe gradiente",
              actor.beta.grad is not None and abs(actor.beta.grad.item()) > 0)


def test_speed():
    actor = QuantumActor(Config())
    x = torch.rand(20, 4)

    t0 = time.time()
    for _ in range(10):
        actor(x)
    lote = (time.time() - t0) / 10

    t0 = time.time()
    for _ in range(10):
        for i in range(20):
            actor(x[i])
    uma_a_uma = (time.time() - t0) / 10

    print(f"       lote de 20:  {lote * 1000:7.1f} ms")
    print(f"       20 chamadas: {uma_a_uma * 1000:7.1f} ms")
    print(f"       ganho:       {uma_a_uma / lote:.1f}x")
    check("lote é mais rápido que o laço", lote < uma_a_uma)


if __name__ == "__main__":
    print("=== circuito ===")
    test_circuit_shapes()
    test_batch_equals_loop()
    test_expval_range()

    print("\n=== política ===")
    test_policy_is_distribution()
    test_normalization()
    test_param_counts()

    print("\n=== gradientes ===")
    test_gradients_flow_through_circuit()

    print("\n=== desempenho ===")
    test_speed()

    print("\n=== circuito desenhado (conferir contra a Fig. 1) ===")
    print(draw())
    print("\nTodos os testes passaram.")
