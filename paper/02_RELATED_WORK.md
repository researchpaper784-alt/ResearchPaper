# 2. Related work

> **Every `[CITE]` in this file must be resolved against the actual paper by a human.** The
> implementation plan names these works without bibliographic detail and this repository has no
> literature access. Inventing an author, year or venue is the fastest route to a desk reject.
> `grep -rn "\[CITE" paper/` lists them all.

## 2.1 Adaptive aggregation weights

FedAvg's size-proportional weighting is the baseline every method here replaces. Four families
have been proposed.

**Analytically derived weights.** Choose $\alpha$ to minimize a convergence upper bound. The
appeal is that the weights follow from the analysis rather than from a heuristic; the cost is
that the bound requires smoothness and bounded gradient dissimilarity, neither of which holds
for the deep CNNs actually deployed. [CITE: FedAAW — confirm name, venue, year, and whether
the bound is on the global objective or on drift]

**Learned weights.** Treat $\alpha$ as parameters and gradient-learn them on a server-held
proxy validation set. This is the strongest prior family and the closest comparison to our
work, so FedLAW is a baseline in §5.1 rather than only a citation. Its requirement is also its
limitation: the server must hold representative data, which in cross-silo medical federation is
frequently the constraint that motivated federating in the first place. [CITE: FedLAW]

**Heuristic weights.** Weight by local loss, or by class contribution. Cheap and needs no
server data, but greedy and effectively one-dimensional — a single scalar signal per client
ordered monotonically. [CITE: FedNolowe; also the class-contribution weighting line]

**RL-optimized weights.** Learn a weighting policy with reinforcement learning. Expressive, but
sample-hungry — many episodes to train a policy — and the policy transfers poorly to a
federation with different client composition. [CITE: DaWa]

Our setting differs from all four in what it assumes: no server-side data (unlike learned), no
convexity or dissimilarity bound (unlike analytic), more than one dimension of client signal
(unlike heuristic), and no cross-deployment policy training (unlike RL).

## 2.2 Swarm intelligence in federated learning

Swarm methods appear in federated learning almost entirely at the **systems layer** rather than
the aggregation layer:

- **ACO for communication scheduling** — which clients transmit updates when, to reduce
  contention or energy. [CITE]
- **PSO for device selection** — choosing the participating subset per round. [CITE]
- **ACO for federated feature selection** — a discrete search over features, run federatedly.
  [CITE]
- **Swarm methods for communication efficiency** more broadly. [CITE]

In each the swarm optimizes *who participates* or *what is transmitted*. None optimizes the
aggregation weight vector itself, and none carries pheromone across communication rounds as a
persistent record of client reliability. That combination — a discrete metaheuristic over
$\alpha$, with cross-round stigmergy — is the gap.

Whether cross-round pheromone memory is the part that earns its keep is an empirical question
and we answer it directly: ablation A2 compares no persistence, decayed persistence and full
persistence. It is also the one capability an equal-budget stateless search *structurally*
cannot provide, which matters for how §5.2's result should be read.

## 2.3 Name collision

An unrelated 2025 paper uses the acronym "FedACo". This work was renamed **FedSwarm** to avoid
the collision; earlier internal documents and some code paths still say "FedACO". The two are
unrelated and the difference should be stated in a footnote rather than left for a reviewer to
discover. [CITE: the 2025 FedACo paper, to cite-and-differentiate]

## 2.4 Robustness and Byzantine aggregation

Krum, Trimmed-Mean and coordinate-wise median are the standard robust aggregators and are
baselines in §5.3. [CITE: Krum; Trimmed-Mean/median] A weighting method that inspects per-client
updates has the *opportunity* to down-weight adversaries as a side effect — whether it does is
measured in §5.3, not assumed. It also has the same limitation as those methods: it requires
per-client updates at the server and so is incompatible with plain secure aggregation (§6.2).
