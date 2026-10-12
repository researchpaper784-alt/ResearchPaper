# 2. Related work

> **Citations resolved 2026-10-11** against a live literature search (Consensus/Semantic
> Scholar); see `paper/README.md`'s "Citations" section and `paper/REFERENCES.md` for the full
> metadata and URLs. The search was automated, not peer-reviewed judgment, so this section is
> still worth a human pass before submission — in particular the "FedAAW" name collision flagged
> inline and in `REFERENCES.md`.

## 2.1 Adaptive aggregation weights

FedAvg's size-proportional weighting is the baseline every method here replaces. Four families
have been proposed.

**Analytically derived weights.** Choose $\alpha$ to minimize a convergence upper bound. The
appeal is that the weights follow from the analysis rather than from a heuristic; the cost is
that the bound requires smoothness and bounded gradient dissimilarity, neither of which holds
for the deep CNNs actually deployed. (Li et al., *Federated Learning With Adaptive Aggregation
Weights for Non-IID Data in Edge Networks* [FedAAW], IEEE TCCN 2025 — the bound is on the
global objective, via a closed-form weight expression that minimizes it. ⚠️ A second, unrelated
paper also uses the name FedAAW, see `paper/REFERENCES.md`.)

**Learned weights.** Treat $\alpha$ as parameters and gradient-learn them on a server-held
proxy validation set. This is the strongest prior family and the closest comparison to our
work, so FedLAW is a baseline in §5.1 rather than only a citation. Its requirement is also its
limitation: the server must hold representative data, which in cross-silo medical federation is
frequently the constraint that motivated federating in the first place. (Li et al., *Revisiting
Weighted Aggregation in Federated Learning with Neural Networks* [FedLAW], 2023.)

**Heuristic weights.** Weight by local loss, or by class contribution. Cheap and needs no
server data, but greedy and effectively one-dimensional — a single scalar signal per client
ordered monotonically. (Le et al., *FedNolowe: A normalized loss-based weighted aggregation
strategy for robust federated learning in heterogeneous environments*, PLOS One 2025 — weights
by normalized training loss, two-stage L1 normalization.)

**RL-optimized weights.** Learn a weighting policy with reinforcement learning. Expressive, but
sample-hungry — many episodes to train a policy — and the policy transfers poorly to a
federation with different client composition. (Sai et al., *Optimizing Hierarchical Federated
Learning: A Reinforcement Learning Approach* [DaWa], IEEE Transactions on Consumer Electronics
2025 — a DRL-trained client-selection-and-weighting policy for a three-layer hierarchical FL
setup.)

**Client-vector weights — the closest prior work.** FedAWA optimizes aggregation weights from
client update vectors, with no proxy dataset, up-weighting clients whose updates align with the
global optimization direction. [Shi et al., CVPR 2025 — metadata in `paper/REFERENCES.md`]

An earlier draft of this section differentiated our work on four axes: no server-side data, no
convexity or dissimilarity bound, more than one dimension of client signal, and no
cross-deployment policy training. **FedAWA satisfies the first two and arguably the third**, so
that framing is withdrawn. The honest distinctions are narrower:

| | FedAWA | this work |
|---|---|---|
| how $\alpha$ is obtained | gradient descent on client vectors | population metaheuristic over a discretized level set |
| state across rounds | **none** | pheromone persists as a record of client reliability |
| objective | update-direction alignment | alignment + dispersion + concentration penalty |

The second row is the one that matters, and it is the one we test directly: ablation A2 compares
no persistence, decayed persistence and full persistence. If persistence does not pay, the
distinction from FedAWA is a search-algorithm substitution and the paper should say so.

## 2.2 Swarm intelligence in federated learning

Swarm methods appear in federated learning almost entirely at the **systems layer** rather than
the aggregation layer:

- **ACO for communication scheduling** — which clients transmit updates when, to reduce
  contention or energy. (Rjoub et al., *A Hybrid Swarm Intelligence Approach for Optimizing
  Multimodal LLM Deployment in Edge-Cloud-based Federated Learning Environments*, Computer
  Communications 2025 — ACO optimizes transmission of model updates.)
- **PSO for device selection** — choosing the participating subset per round. (Same Rjoub et
  al. 2025: PSO selects edge devices in the same pipeline.)
- **ACO for federated feature selection** — a discrete search over features, run federatedly.
  (Sun et al., *Federated multi-label feature selection via manifold sparse constraints and
  game-theoretic evolutionary ant colony optimization* [Fed-MGACO], J. King Saud University CIS
  2025.)
- **Swarm methods more broadly, at the client-selection layer.** (Khan et al., *Swarm
  Intelligence-Driven Client Selection for Federated Learning in Cybersecurity Applications*,
  2024 — benchmarks nine swarm-intelligence algorithms, including ACO and PSO, for client
  selection; finds Grey Wolf Optimization beating both, a second independent signal that ACO is
  not the strongest swarm choice in FL settings generally.)

These operate on *who participates* or *what is transmitted*. **But swarm optimization of the
aggregation weight vector itself is also already published**, and an earlier draft of this
section wrongly asserted otherwise:

- **Adp-FL-PSO** applies an enhanced PSO *at the server to compute the optimal aggregation
  weights* under non-IID data. [Srinivas et al., NMITCON 2025]
- **FedPSO** replaces FedAvg's weight aggregation with PSO, targeting communication cost.
  [Park et al., Sensors 2021]

So "a swarm metaheuristic over $\alpha$ is unexplored" is false, and we do not claim it. The
remaining structural gap is narrower: **cross-round stigmergy**. FedAWA, Adp-FL-PSO and FedPSO
are all stateless between rounds — each recomputes $\alpha$ from the current round's updates. A
persistent per-client memory carried across communication rounds has no analogue in any of them,
and it is the one thing an equal-budget stateless search cannot provide by construction.

That makes ablation A2 — persistence off / decayed / full — the experiment that decides whether
this work has a contribution at all, rather than a secondary ablation.

**A note on what the literature's choice of algorithm implies.** Published swarm work on
aggregation weights uses **PSO**, not ACO. A benchmark of nine swarm algorithms for FL client
selection found Grey Wolf Optimization outperforming both ACO and PSO. [Khan et al., 2024] Our
own equal-budget comparison (§5.2) measures PSO at +0.0484 against ACO at +0.0013–0.0066 on
identical fitness and budget. These are independent signals pointing the same way, and we report
ours as corroborating rather than contradicting the literature — which is a stronger position
than the one this project set out to occupy.

Whether cross-round pheromone memory is the part that earns its keep is an empirical question
and we answer it directly: ablation A2 compares no persistence, decayed persistence and full
persistence. It is also the one capability an equal-budget stateless search *structurally*
cannot provide, which matters for how §5.2's result should be read.

## 2.3 Name collision

An unrelated 2025 paper uses the acronym "FedACo". This work was renamed **FedSwarm** to avoid
the collision; earlier internal documents and some code paths still say "FedACO". The two are
unrelated and the difference should be stated in a footnote rather than left for a reviewer to
discover. (The collision: *FedACo: Adaptive Collaboration with Fine-Grained Aggregation for
Personalized Federated Learning*, PRCV 2025 / Springer,
https://link.springer.com/chapter/10.1007/978-981-95-4987-0_1 — personalization via
fine-grained layer collaboration; shares the acronym by coincidence, not the method.)

## 2.4 Robustness and Byzantine aggregation

Krum, Trimmed-Mean and coordinate-wise median are the standard robust aggregators and are
baselines in §5.3. (Krum: Blanchard et al., *Machine Learning with Adversaries: Byzantine
Tolerant Gradient Descent*, NeurIPS 2017, 3,028 citations — $O(n^2(d+\log n))$ time complexity,
worth stating next to our own $O(K^2d)$ Gram precompute. Trimmed-Mean: Wang et al., *Federated
learning framework based on trimmed mean aggregation rules*, Expert Systems with Applications
2025.) A weighting method that inspects per-client
updates has the *opportunity* to down-weight adversaries as a side effect — whether it does is
measured in §5.3, not assumed. It also has the same limitation as those methods: it requires
per-client updates at the server and so is incompatible with plain secure aggregation (§6.2).
