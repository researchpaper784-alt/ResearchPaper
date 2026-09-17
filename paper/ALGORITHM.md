# FedACO / FedSwarm — algorithm

Pseudocode for `strategies/fedaco.py`'s `FedACO.aggregate_train`, kept in sync with the
code by construction: every numbered step below names the function in `fedswarm/aco/`
that implements it. If this file and the code ever disagree, the code (and its tests)
is ground truth; fix this file, not the other way around.

## Setting

Round $t$, participating clients $S_t$, $|S_t| = K$. Client $k$ returns $\Delta_k =
w_k^{t+1} - w_t$. FedAvg fixes $\alpha_k = n_k / \sum_j n_j$; FedACO searches for
$\alpha \ge 0$, $\sum_k \alpha_k = s$ (fixed shrinkage, `FedACOConfig.target_sum`) each
round, via ant colony optimization over a discretized multiplier grid.

## Per-round pseudocode

```
Input: global state w_t, client replies {(Δ_k, n_k, loss_before_k, loss_after_k)}_{k in S_t}

1.  deltas[K, d]           ← flatten_state_dicts(w_t, client states)        [aco/gram.py]
2.  G[K, K]                ← compute_gram_matrix(deltas)                    [aco/gram.py]
    robust_mean[d]         ← trimmed_mean(deltas, trim_fraction)            [aco/gram.py]
    g_rob[K], rob_norm_sq  ← <deltas, robust_mean>, ||robust_mean||^2       [aco/gram.py precompute_gram]
3.  d_k[K]                 ← sigmoid(β1*a_k - β2*|log r_k| + β3*v_k + β4*q_k)
                              a_k, r_k from G; v_k = loss_before_k - loss_after_k; q_k = n_k/max_j n_j
                                                                              [aco/heuristics.py desirability_scores]
    eta[K, L]               ← (1 + |λ_l - d_hat_k|)^-1                       [aco/heuristics.py desirability_matrix]
4.  tau0[K, L]              ← Pheromone.begin_round(client_ids)              [aco/pheromone.py]
                              (cross-round persistence, keyed by client id -- §4.3)
5.  A_t, I_t                ← colony_budget(round_idx, num_rounds)           [aco/schedules.py]
6.  for iteration in 1..I_t:
        for ant in 1..A_t:
            for station k in 1..K:                                          [aco/colony.py _select_levels]
                with prob q0:  l* = argmax_l tau[k,l]^a * eta[k,l]^b
                else:          l  ~ p(l|k) ∝ tau[k,l]^a * eta[k,l]^b
            alpha_tilde       ← λ[l*] * base_weights                        [aco/colony.py _levels_to_alpha]
            alpha             ← alpha_tilde * target_sum / sum(alpha_tilde)
            fitness           ← F(alpha)                                    [aco/fitness.py DataFreeFitness]
        tau ← (1-ρ)*tau; deposit iteration-best (+ global-best every T iters); clip [tau_min, tau_max]
        stop early if pheromone entropy < threshold or global best stagnant for p iterations
7.  alpha_best, F_best       ← best (alpha, fitness) found across the run     [aco/colony.py ColonyResult]
8.  F_fedavg                 ← F(base_weights * target_sum)
    if F_best <= F_fedavg:  alpha_final ← base_weights * target_sum; fallback_used ← True
    else:                    alpha_final ← alpha_best;                fallback_used ← False
                                                                              [strategies/fedaco.py safety fallback, §4.7]
9.  Pheromone.end_round(client_ids, tau_final)                               [aco/pheromone.py]
10. w_{t+1} ← w_t + Σ_k alpha_final[k] * Δ_k                                 [aco/gram.py apply_delta]

Output: w_{t+1}, metrics {alpha, alpha_entropy, alpha_max, fallback_used, best_fitness,
        fedavg_fitness, pheromone_entropy, aco_time_ms, gram_time_ms, ...}
```

## Fitness (data-free, default — `aco/fitness.py::DataFreeFitness`)

$$F(\alpha) = \gamma_1 \cdot \text{align}(\alpha) - \gamma_2 \cdot \text{disp}(\alpha) - \gamma_3 \cdot (\log K - H(\alpha))$$

computed entirely from $G$, $g_{\text{rob}}$, and $\|\bar\Delta_{\text{rob}}\|^2$
(each precomputed once per round, `aco/gram.py::precompute_gram`) — every per-ant
evaluation is $O(K)$, independent of model size $d$. Two other modes exist behind the
same `Fitness.evaluate(alpha) -> float` interface (`aco/fitness.py`):
`ServerValFitness` (materializes a candidate model, evaluates macro-F1 on a server val
set — an upper-bound reference) and `ClientProbeFitness` (scores a fixed menu of
candidates from client-reported losses collected one round earlier). Both are
implemented and independently testable now; only `data_free` is wired into
`FedACO.aggregate_train`'s normal round loop — `server_val`/`client_probe` are for the
Phase 7 A3 ablation at a heavily reduced colony budget (plan §4.4).

## What is a deliberate scope reduction, not an oversight

- **Global shrinkage $s$** (plan §4.1) is a fixed `FedACOConfig.target_sum`, not an
  extra per-ant search dimension. Ablating different fixed values of $s$ is Phase 7
  work; searching $s$ per-round is not implemented. See `docs/OPEN_QUESTIONS.md`.
- **`client_probe`'s broadcast/collect round-trip** (candidate menu → client-reported
  losses one round later) is not wired into `FedACO`'s `configure_train`/
  `aggregate_train` yet — `ClientProbeFitness` implements the aggregation side only.
  Needed for the Phase 7 A3 ablation, not for the default `data_free` path this file
  otherwise describes.

## Acceptance tests (plan §4.8, all passing — `tests/test_gram.py`, `tests/test_aco.py`, `tests/test_strategy.py`)

1. `test_gram_equivalence` — Gram-computed fitness matches a naive full-vector
   computation to within 1e-4.
2. `test_fedavg_recoverable` — pheromone uniform (guaranteed on round 1), `q0=1`, eta
   forced to peak at $\lambda=1$ ⟹ FedACO's aggregated weights equal plain FedAvg's, to
   float tolerance.
3. `test_planted_bad_client` — a client whose update is a negated, amplified copy of
   the consensus direction gets a weight materially below its FedAvg share.
4. `test_pheromone_persists` — pheromone at round $t{+}1$ is a decayed function of
   round $t$, keyed by client id under a changing participation set.
5. `test_simplex` — returned $\alpha$ is non-negative and sums to `target_sum` within
   1e-6 (including the degenerate all-$\lambda{=}0$ corner case, which falls back to
   the FedAvg point rather than returning an invalid off-simplex vector — a real bug
   this test caught, fixed in `aco/colony.py::_levels_to_alpha`).
6. `test_overhead` — ACO wall-clock is under 5% of a representative local-training
   wall-clock in the test's synthetic smoke-sized setup.
