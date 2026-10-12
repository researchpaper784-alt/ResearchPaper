# 1. Introduction

## 1.1 The problem

FedAvg forms the global model as a fixed, data-size-proportional convex combination of client
updates:

$$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_j n_j} w_k^{t+1}$$

Under non-IID client data these weights are suboptimal. In multi-hospital medical imaging —
the setting this paper targets — scanner vendor, field strength, acquisition protocol and case
mix all differ between sites, so client updates point in conflicting directions. A client whose
update opposes the consensus still contributes in proportion to its dataset size, dragging the
global model and slowing convergence (Li et al., *Federated Optimization in Heterogeneous
Networks*, MLSys 2020; Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning*, ICML 2020 — SCAFFOLD names this failure mode "client drift" directly).

The weight vector $\alpha$ is therefore a natural thing to optimize rather than fix.

## 1.2 Where the literature leaves a gap

Work on adaptive aggregation weights splits four ways:

- **Analytically derived** — minimize a convergence upper bound. Elegant, but the bound rests
  on smoothness and bounded-dissimilarity assumptions that do not hold for deep CNNs.
  (Li et al., *Federated Learning With Adaptive Aggregation Weights for Non-IID Data in Edge
  Networks*, IEEE TCCN 2025 — ⚠️ a second, unrelated paper shares the FedAAW name, see
  `paper/REFERENCES.md`.)
- **Learned** — gradient-learn the weights on a server-side proxy validation set. Strong, but
  requires the server to hold data, which is often the exact thing federation exists to avoid.
  (Li et al., *Revisiting Weighted Aggregation in Federated Learning with Neural Networks*
  [FedLAW], 2023.)
- **Heuristic** — loss-based or class-contribution weighting. Cheap, but greedy and
  one-dimensional. (Le et al., *FedNolowe: A normalized loss-based weighted aggregation
  strategy for robust federated learning in heterogeneous environments*, PLOS One 2025.)
- **RL-optimized** — powerful, but needs many episodes to train a policy and transfers poorly
  across deployments. (Sai et al., *Optimizing Hierarchical Federated Learning: A
  Reinforcement Learning Approach* [DaWa], IEEE TCE 2025.)

Swarm intelligence has appeared in federated learning, but at the **systems layer**: ACO for
scheduling model-update transmission and PSO for edge-device selection (Rjoub et al., *A
Hybrid Swarm Intelligence Approach for Optimizing Multimodal LLM Deployment in Edge-Cloud-based
Federated Learning Environments*, Computer Communications 2025), and ACO for federated
*feature* selection (Sun et al., *Federated multi-label feature selection via manifold sparse
constraints and game-theoretic evolutionary ant colony optimization*, J. King Saud University
CIS 2025).

⚠️ **The gap is substantially narrower than this project assumed, in two ways.** First, swarm
optimization of the aggregation weight vector is *already published*: Adp-FL-PSO applies PSO at
the server to compute optimal aggregation weights [Srinivas et al., 2025], and FedPSO replaces
FedAvg's aggregation with PSO [Park et al., 2021]. Second, FedAWA (Shi et al., CVPR 2025)
adaptively optimizes aggregation weights from *client update vectors*, requires no proxy data,
and up-weights clients whose updates align with the global direction — the same signal as the
alignment term in our objective. The claim "applying a metaheuristic to the aggregation weight
vector is the open slot" does not survive contact with it. What remains genuinely distinct is
narrower and we state it as such: (i) FedAWA optimizes $\alpha$ by gradient descent, we search
it with a population metaheuristic over a discretized level set; (ii) FedAWA is **stateless
across rounds**, and cross-round pheromone persistence has no analogue in it; (iii) our
objective adds a dispersion term and a concentration penalty to alignment rather than using
alignment alone.

Point (ii) is the load-bearing one — **every** published method here (FedAWA, Adp-FL-PSO,
FedPSO) recomputes $\alpha$ from the current round alone — which makes ablation A2, persistence
off / decayed / full, the experiment carrying the novelty claim rather than a secondary ablation.

It is also worth noting which algorithm the prior work chose: **PSO, not ACO.** Our equal-budget
comparison finds PSO ahead of ACO by a wide margin on identical fitness (§5.2), and a published
benchmark of nine swarm algorithms for FL client selection found Grey Wolf Optimization ahead of
both. Our negative result on ACO is consistent with the literature rather than at odds with it.
Full metadata for every reference in this paragraph is in `paper/REFERENCES.md`.

## 1.3 What this paper reports

> **Updated 2026-10-11: ablation A1 has now run on real data (80/80 cells, 8 seeds,
> `configs/experiment/ablation_a1_reduced.yaml`), and it does not reverse this framing — it
> sharpens it.** The three synthetic-fixture screens (below, item 2) found the colony losing to
> every equal-budget control. The real-data run finds something adjacent but distinct: under
> real heterogeneity, *no* equal-budget search method beats the FedAvg point, ACO included but
> not uniquely so (full numbers: §5.2, §6.1). The alternative framing at the end of this file,
> kept for the record, does not apply — the negative framing is the one the real-data evidence
> supports, not only the synthetic one.

We set out to test whether an ant-colony metaheuristic over the aggregation weight vector
outperforms fixed and learned weighting. We report four findings, of which the second and third
are negative and, we argue, more useful than a positive first finding would have been.

1. **A surrogate fitness for aggregation weights admits a degenerate optimum, and standard
   diagnostics do not reveal it.** On real data the best single-client vertex — put all weight
   on one client, discard the rest — outscored the FedAvg reference point in **15 of 15
   rounds** (mean margin +0.6252). The search did not reach that optimum only because it was
   too short to; every health signal we logged read normal while the objective pointed
   somewhere useless. The failure mode arrives as the *search improves*. We give a closed-form
   condition for when the penalty term is large enough to remove the degeneracy.

2. **At equal evaluation budget, the colony does not beat simpler search — on a synthetic
   heterogeneous fixture or on real data.** On the synthetic fixture, coordinate grid search,
   PSO, a GA and uniform random search all gained more. On real data (§5.2), the picture
   changes: under heterogeneity, none of the five methods — ACO included — beats the FedAvg
   point; ACO sits in the worse half of the five, close to PSO and random, well behind GA and
   coordinate-grid. We identify a mechanism for the synthetic result that is not about ant
   colony optimization specifically: the per-client desirability signal spans ~0.04 against a
   level spacing of 0.25, so every client's argmax lands on the same level; a uniform level
   assignment renormalizes to the size-proportional weights exactly; so the greedy branch of
   the ACS construction rule reproduces **the FedAvg point** 70% of the time at $q_0 = 0.7$.
   The colony is anchored to the reference it is meant to improve on; the real-data result shows
   this is not only an ACO problem, under this fitness and this search-budget, for the simpler
   methods tested too.

3. **Cross-round pheromone persistence is the one place the method does something its closest
   published prior work cannot, and on real data it does.** FedAWA, Adp-FL-PSO and FedPSO — the
   published work closest to this method (§2.1–2.2) — are all stateless across federation
   rounds. Ablation A2 (5 seeds, two heterogeneous regimes) tests persistence directly: full
   persistence beats both no persistence and the method's own `decayed` default in both regimes
   tested, by a wide margin at the more heterogeneous of the two. This is a genuine positive
   result, not hedged like items 1–2, though at 5 seeds it is an effect size rather than a
   significance claim (§6.5).

4. **The server-side cost is negligible.** The Gram-matrix formulation makes a fitness
   evaluation $O(K)$ after an $O(K^2 d)$ precompute, measured at **2.6% of round wall-clock**
   at $K=10$. The cost claim holds; the $O(K^2)$ *shape* is not visible over the measured range,
   and we say so rather than fitting a curve to it.

**Contributions.** (i) A reusable diagnostic for search-based aggregation-weight methods —
`corner_margin`, the fitness gap between the best single-client vertex and the reference point
— which detects the degeneracy above from quantities a round already computes. (ii) A
closed-form expression for the penalty weight that removes it. (iii) An equal-budget comparison
that isolates *which* search method matters, with the anchoring mechanism identified, run on
both a synthetic fixture and real data. (iv) The first real-data measurement of whether
cross-round persistence pays for search-based aggregation weighting, isolating it from every
other published method in this space. (v) A component-level pseudo-patient split. Image-level
de-duplication of this dataset is already published; we show it still leaves 7.9% of retained
images chain-linked to another retained image and split independently, and remove that by
splitting on connected components of the near-duplicate graph (§4.2). We also separate a
confound prior work names but does not quantify: the archive's exact class balance is
manufactured by duplicating one class (§4.1).

---

## Alternative framing — did not apply (A1's real-data result is in, 2026-10-11)

**Kept for the record, not deleted, per this file's own original instruction** ("the switch is
a swap, not a rewrite") — but the swap is not needed. A1 on real data did not separate the
colony from the controls; §1.3 above is the framing the evidence supports. Delete this section
entirely once a human has confirmed §1.3's real-data numbers are read correctly; until then it
documents what would have had to be true for this alternative to apply, and clearly was not.

Delete §1.3 above and use this. Do **not** run both.

If the colony separates from all four controls on real data at 8 seeds, the paper is the one
originally planned: ACO-searched aggregation weights beat fixed, heuristic and learned
weighting under non-IID partitions (C1); the gain is attributable to ACO specifically rather
than to weight optimization in general (C2); and the overhead is negligible (C3). §5.1 carries
Table 1 and Figures 1–2, §5.2 carries A1 as the central ablation rather than as the result, and
§6 keeps the secure-aggregation incompatibility and the single-dataset limitation.

In that case the degeneracy of §1.3(1) becomes a *method* subsection — the penalty term and its
closed-form threshold are then part of why the method works, not the finding.
