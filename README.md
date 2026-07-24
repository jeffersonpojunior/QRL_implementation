# Replicação — QRL com Circuito Variacional (CartPole)

Replicação de **Kwak, Yun, Jung, Kim & Kim (2021)**, *Introduction to Quantum
Reinforcement Learning: Theory and PennyLane-based Implementation*
([arXiv:2108.06849](https://arxiv.org/abs/2108.06849)).

Um agente actor-critic em que **o ator é um circuito quântico variacional (VQC)
de 4 qubits** e o crítico é uma MLP clássica, treinado com PPO no CartPole.

---

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python test_circuit.py                    # sanidade — rode ANTES do treino
python train.py                           # configuração recomendada
python train.py --paper                   # hiperparâmetros literais do artigo
python train.py --seed 1 --episodes 1500  # outra seed
python evaluate.py runs/baseline_seed0 --greedy --random   # testa o agente treinado
python plot_results.py runs/baseline_seed* --random-baseline 22
```

---

## Arquivos

| Arquivo | Papel | Corresponde a |
|---|---|---|
| `qrl.py` | hiperparâmetros, circuito, ator, crítico e buffer | Seção IV-C, **Figs. 1 e 2** |
| `train.py` | loop de coleta + atualização PPO | **Algorithm 1** |
| `evaluate.py` | testa a política congelada, pós-treino | — |
| `plot_results.py` | figura de resultados | **Fig. 4** |
| `test_circuit.py` | testes de sanidade | — |

`python qrl.py` desenha o circuito em ASCII, para conferir contra a Fig. 1.

### A integração acontece em uma linha

```python
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights): ...
```

O decorador transforma o circuito numa função PyTorch diferenciável. A partir
daí `W` é um `nn.Parameter` comum e o `Adam` não sabe que aqueles 48 números
são ângulos de portas quânticas. É por isso que o resto do algoritmo é PPO
padrão, sem adaptação nenhuma.

---

## Desvios em relação ao artigo — e por quê

Cada item abaixo está marcado no `config.py`. Para reproduzir o artigo ao pé
da letra, use `python train.py --paper`.

### 1. Amostragem da política em vez de ε-greedy

O Algorithm 1 seleciona ações com ε-greedy sobre `max_a Q*(s,a;θ)` e calcula o
alvo TD com `max_a' Q` — vocabulário de **DQN**. Mas também calcula a razão
`r_j = π_θ / π_θ_OLD` e a perda clipada — vocabulário de **PPO**. As duas
interpretações da saída do circuito são incompatíveis: ou são valores-Q, ou é
uma distribuição de probabilidade.

A Fig. 2 rotula a seta que sai do ator quântico como **π_θ(a|s)**, e a razão
do PPO exige política estocástica. Seguimos a figura: amostramos de π e
descartamos o ε-greedy.

### 2. Buffer on-policy, esvaziado a cada atualização

PPO é on-policy. O Algorithm 1 manda amostrar mini-batches aleatórios de uma
"replay memory D de capacidade N", o que é herança do DQN. Reamostrar
transições antigas faz a razão sair permanentemente da faixa de clipping e o
gradiente morre. `rollout.py` descarta o buffer após cada `make_batch()`.

### 3. `lr_critic` de 1e-5 para 1e-3

O artigo usa **1e-3 para o ator e 1e-5 para o crítico** — o crítico aprende
100× mais devagar. Em actor-critic isso é ao contrário do usual: V(s) precisa
convergir rápido para que a vantagem Â seja informativa. Com 1e-5 o crítico
praticamente não sai da inicialização, as vantagens iniciais são ruído, e isso
é candidato a explicar boa parte da variância enorme reportada na Fig. 4.

**Vale rodar os dois e comparar** — é um resultado interessante para a
apresentação.

### 4. `eps_clip` de 0.01 para 0.2

O artigo usa ε = 0.01, um clipping dez a vinte vezes mais apertado que o
padrão da literatura (0.1–0.2). Isso limita muito o passo por atualização.

### 5. Temperatura `beta` treinável

`<PauliY>` vive em [-1, 1]. Um softmax direto sobre esse intervalo produz uma
política quase uniforme (razão máxima e² ≈ 7.4), o que trava o aprendizado no
começo. Adicionamos um escalar treinável multiplicando a saída antes do
softmax — um parâmetro a mais (49 em vez de 48). Desative com
`trainable_beta=False`.

### 6. Bônus de entropia e normalização da vantagem

Duas práticas padrão de PPO que o artigo não menciona. Ambas configuráveis.

### 7. `CartPole-v1` em vez de `v0`

`v0` satura em 200 e `v1` em 500, dando mais espaço para ver a curva. Use
`--paper` para voltar ao v0.

---

## Detalhes de implementação que importam

**Normalização da observação.** O artigo diz que as entradas são normalizadas
entre −π e π, sem especificar como. Usamos `clamp(obs / [2.4, 2.5, 0.21, 2.5],
-1, 1)` e o circuito multiplica por π. As velocidades não têm limite formal no
CartPole, daí a saturação.

**Broadcasting.** O código de 2021 executa o circuito para um estado por vez.
Versões atuais do PennyLane suportam *parameter broadcasting*: passe
`inputs` com shape `(B, 4)` e todo o mini-batch é simulado de uma vez.
Medido aqui: **~16× mais rápido** que o laço. `test_circuit.py` verifica que
lote e laço dão resultados idênticos.

**`diff_method`.** Com `default.qubit` + `backprop`, o gradiente é obtido por
retropropagação através da simulação do vetor de estado — barato. Em hardware
real cai para `parameter-shift`, que custa **2 execuções extras por
parâmetro**: 96 execuções por amostra. Se trocar de device, recalcule o
orçamento de tempo.

**O circuito roda duas vezes por transição.** Uma na coleta (para agir) e
outra na atualização (para ter a razão `r_j` dentro do grafo de autograd). A
π guardada no buffer é `π_θ_OLD` e está desconectada do grafo — não dá para
reaproveitar.

**API antiga.** O código do artigo usa `torch.autograd.Variable`, deprecado
há anos. Use `torch.tensor(..., requires_grad=True)` ou `nn.Parameter`.

---

## Protocolo sugerido para o relatório

```bash
for seed in 0 1 2 3 4; do
    python train.py --seed $seed --episodes 1500 --run-name baseline
    python train.py --seed $seed --episodes 1500 --paper
done
python plot_results.py runs/baseline_seed* --random-baseline 22 --out fig_baseline.png
python plot_results.py runs/paper_faithful_seed* --random-baseline 22 --out fig_paper.png
```

Reportar **média e desvio sobre as seeds**, não uma curva única. Como o achado
central do artigo é justamente a alta variância, uma execução só não sustenta
nenhuma conclusão.

## Números de referência

| | |
|---|---|
| Parâmetros do ator | 48 ângulos (49 com `beta`) |
| Parâmetros do crítico | 1.537 |
| Baseline aleatória (CartPole) | ~22 |
| Tempo por episódio (CPU, lote de 20) | ~0,3–1,2 s, cresce com o score |
