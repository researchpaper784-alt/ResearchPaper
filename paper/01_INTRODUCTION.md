# 1. Introduction

## 1.1 The problem

FedAvg forms the global model as a fixed, data-size-proportional convex combination of client
updates:

$$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_j n_j} w_k^{t+1}$$

Under non-IID client data these weights are suboptimal. In multi-hospital medical imaging —
the setting this paper targets — scanner vendor, field strength, acquisition protocol and case
mix all differ between sites, so client updates point in conflicting directions. A client whose
update opposes the consensus still contributes in proportion to its dataset size, dragging the
global model and slowing convergence. [CITE: client drift / non-IID convergence, e.g. the
FedProx and SCAFFOLD analyses]

The weight vector $\alpha$ is therefore a natural thing to optimize rather than fix.

## 1.2 Where the literature leaves a gap

Work on adaptive aggregation weights splits four ways:

- **Analytically derived** — minimize a convergence upper bound. Elegant, but the bound rests
  on smoothness and bounded-dissimilarity assumptions that do not hold for deep CNNs.
  [CITE: FedAAW]
- **Learned** — gradient-learn the weights on a server-side proxy validation set. Strong, but
  requires the server to hold data, which is often the exact thing federation exists to avoid.
  [CITE: FedLAW]
- **Heuristic** — loss-based or class-contribution weighting. Cheap, but greedy and
  one-dimensional. [CITE: FedNolowe]
- **RL-optimized** — powerful, but needs many episodes to train a policy and transfers poorly
  across deployments. [CITE: DaWa]

Swarm intelligence has appeared in federated learning, but at the **systems layer**: ACO for
scheduling model-update transmission, PSO for edge-device selection, ACO for federated
*feature* selection. [CITE: each of these three] Applying an ACO metaheuristic directly to the
aggregation weight vector, with pheromone persisting across communication rounds as a memory
of client reliability, is the open slot this paper occupies.

## 1.3 What this paper reports

> ⚠️ **This section is written against the evidence as of 2026-09-25 and must be rewritten once
> ablation A1 runs on real data.** Three screens on a heterogeneous synthetic fixture found the
> colony *losing* to every equal-budget control, not tying it (see §6.1 and
> `docs/EXPERIMENT_LOG.md`). Plan §11's week-5 gate calls for reframing on exactly that result.
> The framing below is the one the current evidence supports. If A1 reverses it, the
> alternative framing is spelled out at the end of this file so the switch is a swap, not a
> rewrite.

We set out to test whether an ant-colony metaheuristic over the aggregation weight vector
outperforms fixed and learned weighting. We report three findings, of which the second is
negative and, we argue, more useful than the first would have been.

1. **A surrogate fitness for aggregation weights admits a degenerate optimum, and standard
   diagnostics do not reveal it.** On real data the best single-client vertex — put all weight
   on one client, discard the rest — outscored the FedAvg reference point in **15 of 15
   rounds** (mean margin +0.6252). The search did not reach that optimum only because it was
   too short to; every health signal we logged read normal while the objective pointed
   somewhere useless. The failure mode arrives as the *search improves*. We give a closed-form
   condition for when the penalty term is large enough to remove the degeneracy.

2. **At equal evaluation budget, the colony does not beat simpler search.** Coordinate grid
   search, PSO, a GA and uniform random search all gained more. We identify the mechanism, and
   it is not about ant colony optimization: the per-client desirability signal spans ~0.04
   against a level spacing of 0.25, so every client's argmax lands on the same level; a uniform
   level assignment renormalizes to the size-proportional weights exactly; so the greedy branch
   of the ACS construction rule reproduces **the FedAvg point** 70% of the time at $q_0 = 0.7$.
   The colony is anchored to the reference it is meant to improve on.

3. **The server-side cost is negligible.** The Gram-matrix formulation makes a fitness
   evaluation $O(K)$ after an $O(K^2 d)$ precompute, measured at **2.6% of round wall-clock**
   at $K=10$. The cost claim holds; the $O(K^2)$ *shape* is not visible over the measured range,
   and we say so rather than fitting a curve to it.

**Contributions.** (i) A reusable diagnostic for search-based aggregation-weight methods —
`corner_margin`, the fitness gap between the best single-client vertex and the reference point
— which detects the degeneracy above from quantities a round already computes. (ii) A
closed-form expression for the penalty weight that removes it. (iii) An equal-budget comparison
that isolates *which* search method matters, with the anchoring mechanism identified. (iv) A
fully de-duplicated, leakage-audited pseudo-patient split for this dataset, on which the
published class balance turns out to be an artifact of duplicating one class (§4.1).

---

## Alternative framing, if A1 comes back positive

Delete §1.3 above and use this. Do **not** run both.

If the colony separates from all four controls on real data at 8 seeds, the paper is the one
originally planned: ACO-searched aggregation weights beat fixed, heuristic and learned
weighting under non-IID partitions (C1); the gain is attributable to ACO specifically rather
than to weight optimization in general (C2); and the overhead is negligible (C3). §5.1 carries
Table 1 and Figures 1–2, §5.2 carries A1 as the central ablation rather than as the result, and
§6 keeps the secure-aggregation incompatibility and the single-dataset limitation.

In that case the degeneracy of §1.3(1) becomes a *method* subsection — the penalty term and its
closed-form threshold are then part of why the method works, not the finding.
