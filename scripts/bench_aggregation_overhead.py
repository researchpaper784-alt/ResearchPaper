"""Claim C3: aggregation overhead is negligible and scales as O(K^2).

A direct microbenchmark of the server-side work -- `precompute_gram` plus `run_colony` on
K x d deltas -- rather than a federation, because `overhead.yaml` goes to K=200, which needs
200 Ray actors and the real dataset, neither of which this container has.

**Why this proxy is valid where the isolated colony probe was not.** That probe ranked search
methods by fitness and failed because its synthetic landscape did not reproduce the real one.
This measures wall-clock of a deterministic computation on arrays of a given shape:
`precompute_gram` on a K x d tensor does identical arithmetic whether the deltas came from
training or from a generator, so the timing curve does not depend on the data.

It is also validated rather than asserted -- the project has one real measurement, 198 ms of
aco_time_ms + gram_time_ms at K=10 on a T4, so K=10 is a checkable anchor. That is the step
the colony probe skipped.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig
from fedswarm.aco.gram import precompute_gram
from fedswarm.aco.heuristics import HeuristicWeights, desirability_matrix, desirability_scores
from fedswarm.aco.pheromone import Pheromone
from fedswarm.aco.schedules import level_set

D = 390_404          # simple_cnn state dict; image-size independent (adaptive pool)
# Aggregation runs on CPU in real runs too: `precompute_gram` builds its output with
# `device=deltas.device`, and the deltas come from Flower message arrays. FedACO's
# `device` argument is used only by `fitness_mode="server_val"`. So a CPU benchmark is
# the right instrument here, not a compromise -- though the host CPU differs from a
# Kaggle T4 box's, so the constant travels less well than the shape.
KS = (5, 10, 20, 50, 100, 150, 200)
REPEATS = 3
LEVELS = level_set(11, 0.0, 2.5)

torch.manual_seed(0)
print(f"d = {D:,}  L = {len(LEVELS)}  repeats = {REPEATS}  torch threads = {torch.get_num_threads()}\n")
print(f"{'K':>5} {'gram ms':>9} {'colony ms':>10} {'total ms':>9} {'total/K^2':>11}")
print("-" * 50)

rows = []
for k in KS:
    deltas = torch.randn(k, D) * 0.01
    num_examples = torch.randint(50, 500, (k,)).float()
    gram_times, colony_times = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        gram = precompute_gram(deltas, trim_fraction=0.2)
        gram_times.append((time.perf_counter() - t0) * 1000)

        base = num_examples / num_examples.sum()
        fitness = DataFreeFitness(gram, DataFreeFitnessConfig())
        d_k = desirability_scores(gram, num_examples, None, HeuristicWeights())
        eta = desirability_matrix(d_k, LEVELS)
        pher = Pheromone(num_levels=len(LEVELS))
        tau0 = pher.begin_round([str(i) for i in range(k)])
        t0 = time.perf_counter()
        run_colony(
            tau0=tau0, eta=eta, levels=LEVELS, base_weights=base,
            fitness_fn=fitness.evaluate, num_ants=30, num_iterations=10,
            config=ColonyConfig(), generator=torch.Generator().manual_seed(0),
        )
        colony_times.append((time.perf_counter() - t0) * 1000)

    g, c = statistics.median(gram_times), statistics.median(colony_times)
    rows.append((k, g, c, g + c))
    print(f"{k:5d} {g:9.1f} {c:10.1f} {g+c:9.1f} {(g+c)/k**2:11.4f}")


def fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Least squares through the origin, as figures.plot_overhead_vs_k does."""
    c = float((x @ y) / (x @ x))
    r2 = 1 - float(((y - c * x) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return c, r2


ks = torch.tensor([float(r[0]) for r in rows])
gram_only = torch.tensor([r[1] for r in rows])
colony_only = torch.tensor([r[2] for r in rows])
tot = torch.tensor([r[3] for r in rows])

c2, r2_2 = fit(ks**2, tot)
c1, r2_1 = fit(ks, tot)
print(f"\ntotal vs K^2:  {c2:.3e} * K^2 ms   R^2 = {r2_2:.4f}")
print(f"total vs K  :  {c1:.3e} * K   ms   R^2 = {r2_1:.4f}   (contrast: K^2 should win)")

cg, r2g = fit(ks**2, gram_only)
cc, r2c = fit(ks, colony_only)
print(f"\ngram alone   vs K^2: {cg:.3e}  R^2 = {r2g:.4f}   (the O(K^2 d) term)")
print(f"colony alone vs K  : {cc:.3e}  R^2 = {r2c:.4f}   (O(A*I*K) via the Gram trick)")

k10 = next(r[3] for r in rows if r[0] == 10)
print(f"\nVALIDATION at K=10: {k10:.0f} ms here vs 198 ms measured on the real T4 run "
      f"(ratio {k10/198:.2f}x).")
print("Different hardware, so the constant is not expected to match; the K^2 shape is what")
print("C3 claims and what the fits above test.")

print("\n--- C3 in the terms the paper needs ---")
for k, g, c, t in rows:
    # 7.5 s/round measured at K=10 on a T4; scaled linearly in K for the training cost,
    # which is what the plan's own extrapolation assumes.
    round_ms = 7500.0 * (k / 10.0)
    print(f"  K={k:4d}: overhead {t:8.1f} ms against a {round_ms:8.0f} ms round "
          f"= {100*t/round_ms:5.2f}% of wall clock")
