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

## Algorithm 1 — FedACO round (`strategies/fedaco.py::FedACO.aggregate_train`)

```
Algorithm 1: FedACO round
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
        for ant in 1..A_t:      (RNG: torch.Generator seeded f(config.seed, server_round))
            for station k in 1..K:                                          [aco/colony.py _select_levels]
                with prob q0:  l* = argmax_l tau[k,l]^a * eta[k,l]^b
                else:          l  ~ p(l|k) ∝ tau[k,l]^a * eta[k,l]^b
            alpha_tilde       ← λ[l*] * base_weights                        [aco/colony.py _levels_to_alpha]
            alpha             ← alpha_tilde * target_sum / sum(alpha_tilde)
            fitness           ← F(alpha)                                    [aco/fitness.py DataFreeFitness]
        tau ← (1-ρ)*tau; deposit ρ*Q*max(F,0) on iteration-best (+ global-best every
             T iters); clip [tau_min, tau_max]      [the max(F,0) floor is why F's
             scale matters -- see the fitness note below]
        stop early if pheromone entropy < threshold or global best stagnant for p iterations
7.  alpha_best, F_best       ← best (alpha, fitness) found across the run     [aco/colony.py ColonyResult]
8.  F_fedavg                 ← F(base_weights * target_sum)
    if F_best <= F_fedavg:  alpha_final ← base_weights * target_sum; fallback_used ← True
    else:                    alpha_final ← alpha_best;                fallback_used ← False
                                                                              [strategies/fedaco.py safety fallback, §4.7]
8.  Pheromone.end_round(client_ids, tau_final)                               [aco/pheromone.py]
9.  w_{t+1} ← w_t + Σ_k alpha_final[k] * Δ_k                                 [aco/gram.py apply_delta]

Output: w_{t+1}, metrics {alpha, alpha_entropy, alpha_max, fallback_used, best_fitness,
        fedavg_fitness, pheromone_entropy, delta_mean_sq_norm, aco_time_ms,
        gram_time_ms, ...}
        (pheromone_entropy pinned at log(L) together with a large delta_mean_sq_norm is
         the signature of the degenerate regime described under Fitness, below --
         fallback_used cannot report it)
```

**A1 ablation note (plan §7, `aco/controls.py`)**: step 6 is only Algorithm 2 when
`search_method="aco"` (the default). Setting `FedACOConfig.search_method` to
`"random"`/`"coordinate_grid"`/`"pso"`/`"ga"` replaces step 6 with the matching
control from `aco/controls.py`, run against the *identical* `Fitness` object,
hard-capped at the *identical* evaluation budget `A_t * I_t` via `BudgetedFitness`
(`aco/fitness.py`) — steps 4 and 8 (pheromone) are skipped entirely for controls,
since none of them have a cross-round-memory concept.

## Algorithm 2 — ant construction (`aco/colony.py::run_colony`)

```
Algorithm 2: ant construction and colony run
Input: tau0[K, L], eta[K, L], levels Λ[L], base_weights[K], Fitness F, A_t, I_t

1.  tau ← tau0; best_alpha ← None; best_fitness ← -inf
2.  for iteration in 1..I_t:
3.      iter_best_fitness ← -inf
4.      for ant in 1..A_t:
5.          for station k in 1..K:                                          [aco/colony.py _select_levels]
                with prob q0:  l* ← argmax_l tau[k,l]^a * eta[k,l]^b         (exploitation)
                else:          l  ← sample p(l|k) ∝ tau[k,l]^a * eta[k,l]^b (exploration)
6.          alpha_tilde ← λ[l*] * base_weights                              [aco/colony.py levels_to_alpha]
            alpha       ← alpha_tilde * target_sum / sum(alpha_tilde)
            (if sum(alpha_tilde) ≈ 0 -- every station picked λ=0 -- fall back to
             base_weights * target_sum instead of an invalid off-simplex vector;
             a real bug this exact path caught, plan §4.8 acceptance #5)
7.          fitness ← F(alpha)                                              [Algorithm 3's F, aco/fitness.py]
8.          track iteration-best (alpha, fitness)
9.      evaporate: tau ← (1-ρ)*tau
10.     deposit iteration-best (+ global-best every T iterations) -- Algorithm 3
11.     clip tau to [tau_min, tau_max]; update best_alpha/best_fitness if improved
12.     stop early if pheromone entropy < threshold, or best hasn't improved for p iterations

Output: best_alpha, best_fitness, tau (for Pheromone.end_round)
```

## Algorithm 3 — pheromone update (`aco/colony.py`'s per-iteration step + `aco/pheromone.py`'s cross-round step)

Two distinct timescales share the name "pheromone update" in the plan and are worth
separating explicitly:

```
Algorithm 3a: within-colony evaporation + deposit (every iteration, inside Algorithm 2)
tau ← (1-ρ) * tau                                    (evaporation, all K*L cells)
deposit ← Q * max(iteration_best_fitness, 0)          (non-negative: F can be negative by
                                                        construction, plan §4.5; a negative
                                                        deposit would corrupt tau's meaning
                                                        as a desirability signal)
tau[k, l_k] += ρ * deposit   for each station k's iteration-best level l_k
every T iterations: also deposit from the global best, the same way
tau ← clip(tau, tau_min, tau_max)                     (MAX-MIN Ant System bound)

Algorithm 3b: cross-round persistence (once per FedACO round, aco/pheromone.py)
for each client k that participated this round:
    tau_k ← (1 - ρ_round) * tau_k_final + ρ_round * tau0     if persistence = "decayed"
    tau_k ← tau_k_final                                       if persistence = "full"
    tau_k ← tau0                                               if persistence = "none"
for each client k that did NOT participate this round:
    tau_k ← tau_k * decay^absent_rounds + tau0 * (1 - decay^absent_rounds)
    (absence decay applies regardless of persistence mode -- "none" vs "decayed" only
     differ in how a *participating* client's row is treated, plan §4.3)
```

## Complexity analysis

Let $d$ = model parameter count, $K$ = participating clients, $A_t, I_t$ = this
round's ant/iteration budget (`aco/schedules.py::colony_budget`, linearly decaying
from $(A_0, I_0)$ to $(A_{\min}, I_{\min})$ over the run).

| Step | Cost | Where |
|---|---|---|
| Flatten client states into deltas | $O(K d)$ | `aco/gram.py::flatten_state_dicts` |
| Gram matrix $G$ | $O(K^2 d)$, one pass | `aco/gram.py::compute_gram_matrix` |
| Trimmed mean + $g_{\text{rob}}$ | $O(K d)$ | `aco/gram.py::precompute_gram` |
| Heuristic desirability $\eta$ | $O(K L)$ | `aco/heuristics.py` |
| **Each colony fitness evaluation** | $O(K)$ — **not** $O(d)$ | `aco/fitness.py::DataFreeFitness`, the Gram trick (§4.6) |
| Full colony run | $O(A_t I_t K)$ | `aco/colony.py::run_colony` |
| Materializing the winning $\alpha$ | $O(K d)$, once | `aco/gram.py::apply_delta` |

**Total per round**: $O(K^2 d)$ (the Gram pass, unavoidable and paid once) $+\ O(A_t
I_t K)$ (the colony search, since every evaluation is $O(K)$) $+\ O(K d)$
(materialization). With $K=20$, $A_0=30$, $I_0=10$: 300 evaluations $\times$ $O(K)$
work $\approx$ negligible next to the $O(K^2 d)$ Gram pass, which is itself smaller
than a single client's local epoch (plan §4.6's own framing) — this is what
`test_overhead` (plan §4.8 acceptance #6) checks holds in practice, and what R5
(client-count scaling, `configs/experiment/overhead.yaml`) is designed to verify
empirically as a measured curve against the $K^2$ term.

`ServerValFitness` (the `server_val` A3 ablation mode) breaks the $O(K)$-per-
evaluation property — each evaluation materializes a real candidate model and runs
a forward pass, $O(d)$ per evaluation, hence the plan's own instruction to run it at
a heavily reduced budget ($A{=}8$, $I{=}3$) rather than the default's $A_0{=}30$,
$I_0{=}10$.

## Default hyperparameters (plan §14, `configs/strategy/fedaco.yaml`)

| Plan §14 name | This repo's flat key | Default |
|---|---|---|
| `levels` | `fedaco-num-levels`, `-level-low`, `-level-high` | 11 levels, log-spaced over [0.0, 2.5] |
| `alpha_exponent_a` | `fedaco-a-exponent` | 1.0 |
| `beta_exponent_b` | `fedaco-b-exponent` | 2.0 |
| `rho_evaporation` | `fedaco-rho` | 0.10 |
| `q0_exploitation` | `fedaco-q0` | 0.70 |
| `n_ants` | `fedaco-ants-start` (decays to `fedaco-ants-end`=10) | 30 |
| `n_iterations` | `fedaco-iters-start` (decays to `fedaco-iters-end`=4) | 10 |
| `tau_init` | `fedaco-tau0` | 1.0 |
| `tau_min` / `tau_max` | `fedaco-tau-min` / `fedaco-tau-max` | 0.01 / 10.0 |
| `pheromone_persistence` | `fedaco-pheromone-persistence` | `decayed` |
| `rho_round` | `fedaco-rho-round` | 0.30 |
| `fitness_mode` | `fedaco-fitness-mode` | `data_free` |
| `gamma1_alignment` | `fedaco-gamma-alignment` | 1.0 |
| `gamma2_dispersion` | `fedaco-gamma-dispersion` | 0.50 |
| `gamma3_entropy` | `fedaco-gamma-entropy` | 0.10 |
| `weight_sum` | `fedaco-target-sum` | 1.0 (fixed, not "learned" — see below) |
| `safety_fallback` | `fedaco-safety-fallback` | `true` |
| `budget_schedule` | *(implementation-fixed: linear, not the plan's suggested cosine)* | n/a |
| `early_stop_patience` | `ColonyConfig.stagnation_patience` (not yet a flat run_config key) | 5 |
| *(not in plan §14)* | `fedaco-search-method` | `aco` — the A1 ablation axis |

Two real deviations from §14, both already documented in `docs/OPEN_QUESTIONS.md`
and in `configs/strategy/fedaco.yaml`'s own comments: the budget schedule is linear
between the start/end (A_0,I_0)/(A_min,I_min) pair, not cosine (a shape difference,
not a different starting point — $A_0{=}30$, $I_0{=}10$ match §14 exactly); and
`weight_sum` is a fixed config value, not a per-round *learned* shrinkage factor
(Phase 7's A7 sweeps fixed values of it, not a learned version).

## Fitness (data-free, default — `aco/fitness.py::DataFreeFitness`)

$$F(\alpha) = \gamma_1 \cdot \text{align}(\alpha) - \gamma_2 \cdot \frac{\text{disp}(\alpha)}{\text{tr}(G)/K} - \gamma_3 \cdot (\log K - H(\alpha))$$

computed entirely from $G$, $g_{\text{rob}}$, and $\|\bar\Delta_{\text{rob}}\|^2$
(each precomputed once per round, `aco/gram.py::precompute_gram`) — every per-ant
evaluation is $O(K)$, independent of model size $d$.

**The $\text{tr}(G)/K$ divisor is load-bearing, not cosmetic** (added 2026-09-17;
`GramPrecompute.mean_sq_norm`). $\text{align}$ is a cosine in $[-1, 1]$ and the entropy
penalty lies in $[0, \log K]$, but $\text{disp}$ is a raw sum of squared distances in
units of $\|\Delta\|^2$, so its magnitude tracks the local-training step size rather
than anything about the *quality* of $\alpha$. Measured on real SimpleCNN deltas over
the $(\text{lr}, \text{local epochs})$ grid Phases 6–7 sweep, $\text{disp}$ at the
FedAvg point ranged $0.40$ to $266.7$ — a 660x swing — which drove $F < 0$ for every
candidate in 6 of 9 configs. Since step 6 deposits $\rho \cdot Q \cdot \max(F, 0)$,
that zeroed every deposit: $\tau$ stayed uniform, $\tau^a \eta^b$ collapsed to
$\eta^b$, and the algorithm silently degenerated to deterministic heuristic-greedy
weighting with no colony search and no cross-round stigmergy. The safety fallback in
step 8 cannot detect this, because both sides of its comparison use the same $F$.
Dividing by $\text{tr}(G)/K$ — a per-round constant, so it cannot reorder candidates,
only reweight the term against $\text{align}$ — holds $\text{disp}$ in $[0.763, 0.933]$
across that same grid. Guarded by `test_fitness_is_scale_invariant_under_normalization`
and `test_pheromone_still_carries_signal_when_updates_are_large`; see
docs/EXPERIMENT_LOG.md (2026-09-17) for the full measurements.

Two other modes exist behind the
same `Fitness.evaluate(alpha) -> float` interface (`aco/fitness.py`):
`ServerValFitness` (materializes a candidate model, evaluates macro-F1 on a server val
set — an upper-bound reference) and `ClientProbeFitness` (scores a fixed menu of
candidates from client-reported losses collected one round earlier).
`fedaco-fitness-mode=server_val` is fully wired into `FedACO.aggregate_train` (pass
`model`/`val_loader`/`device` to the constructor, or let `strategies/factory.py` do it
via `fl/app.py::global_val_loader`) — `client_probe` is the one mode still not wired
for live execution (selecting it raises `NotImplementedError`, not a silent fallback);
see the scope-reduction note below.

## What is a deliberate scope reduction, not an oversight

- **Global shrinkage $s$** (plan §4.1) is a fixed `FedACOConfig.target_sum`, not an
  extra per-ant search dimension. Ablating different fixed values of $s$ is Phase 7
  work; searching $s$ per-round is not implemented. See `docs/OPEN_QUESTIONS.md`.
- **`client_probe`'s broadcast/collect round-trip** (candidate menu → client-reported
  losses one round later) is not wired into `FedACO`'s `configure_train`/
  `aggregate_train` yet — `ClientProbeFitness` implements the aggregation side only.
  Needed for the Phase 7 A3 ablation, not for the default `data_free` path this file
  otherwise describes. `server_val` *is* selectable (`aco-fitness-mode="server_val"`,
  requires model/val_loader/device); `client_probe` is rejected at construction rather
  than silently accepted.

Everything else is reachable from `--run-config` via the `aco-*` keys declared in
`pyproject.toml`'s `[tool.flwr.app.config]` — including `aco-persistence` (the Phase 7
A1 ablation) and `aco-target-sum` (the $s$ sweep above), neither of which had any way in
before 2026-09-17.

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
6. `test_overhead` — ACO wall-clock is under 5% (the plan's bar) of a representative
   local-training wall-clock. Measured at the real primary-config scale (K=20,
   SimpleCNN@112, d=390,404): 145 ms of ACO per round against a 56.8 s sequential
   round, i.e. 0.26%, of which 60 ms is the one-off $O(K^2 d)$ Gram precompute.
7. `test_fitness_is_scale_invariant_under_normalization` — scaling every client update
   by a constant leaves $F$ unchanged (and demonstrably does not, without the
   normalization above).
8. `test_pheromone_still_carries_signal_when_updates_are_large` — $\tau$ develops real
   within-row structure at a delta magnitude that previously flattened it.
9. `test_colony_is_reproducible_and_independent_of_global_rng` — a seeded colony is a
   function of `(seed, server_round)` alone; churning global torch RNG between two
   otherwise-identical runs does not change the result.
