"""When does the data-free fitness prefer "use one client and discard the rest"?

Dispersion is `sum_k alpha_k ||delta_k - Delta(alpha)||^2`, a weighted variance about the
weighted mean. At a single-client vertex `alpha = e_j` the mean *is* `delta_j`, so every
term is zero -- the term vanishes outright, for any data whatsoever. Only the
concentration penalty `gamma_3 * log K` stands against that, and it grows as log K, so the
smaller the federation the weaker the guard.

That makes the degenerate optimum a property of (K, gamma_3, heterogeneity) rather than of
any particular dataset, and it can be mapped before a single real run. What this script
maps is `F(best vertex) - F(FedAvg point)`: positive means a *correctly working* colony
should be expected to return a one-client answer, and every other signal in the health
report -- fallback rate, best-vs-FedAvg fitness, a confident alpha -- will read as success
while it does.

⚠️ The client updates are synthetic: a shared consensus direction plus isotropic Gaussian
noise, with `noise` the heterogeneity knob. Real training deltas are not isotropic, so the
crossover values here are indicative, not measured properties of this dataset. What is
exact and data-independent is the vertex identity above -- `fitness.corner_margin`
computes the real number from a real round's Gram matrix, and it is logged per round.

Run:
    python scripts/analyze_fitness_landscape.py
    python scripts/analyze_fitness_landscape.py --gamma-entropy 0.25
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from fedswarm.aco.fitness import (
    CONCENTRATION_PENALTIES,
    DataFreeFitnessConfig,
    corner_margin,
)
from fedswarm.aco.gram import precompute_gram

CLIENT_COUNTS = (2, 4, 6, 10, 20, 50)
HETEROGENEITY = (0.3, 0.5, 1.0, 2.0, 4.0)
SEEDS = range(5)
DIM = 2000


def synthetic_deltas(num_clients: int, noise: float, seed: int, dim: int = DIM) -> torch.Tensor:
    """A shared consensus direction plus per-client isotropic noise. `noise` is the ratio
    of disagreement to consensus: 0.3 is a near-IID federation, 4.0 is one where the
    common direction is almost buried."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=generator)
    direction /= direction.norm()
    return direction + noise * torch.randn(num_clients, dim, generator=generator) / math.sqrt(dim)


def margin(
    num_clients: int, noise: float, seed: int, gamma_entropy: float, shape: str = "entropy"
) -> float:
    gram = precompute_gram(synthetic_deltas(num_clients, noise, seed))
    return corner_margin(
        gram,
        torch.full((num_clients,), 1.0 / num_clients),
        DataFreeFitnessConfig(gamma_entropy=gamma_entropy, concentration_penalty=shape),
    )


def mean_margin(
    num_clients: int, noise: float, gamma_entropy: float, shape: str = "entropy"
) -> float:
    return statistics.fmean(margin(num_clients, noise, s, gamma_entropy, shape) for s in SEEDS)


def required_gamma_entropy(num_clients: int, noise: float, shape: str = "entropy") -> float:
    """The smallest gamma_3 for which the FedAvg point outscores the best vertex.

    Bisection rather than algebra because the FedAvg side moves with gamma_3 too (its own
    concentration penalty is zero, but keeping the search honest costs nothing at this
    size). Monotone in gamma_3: raising the penalty only ever hurts the vertex more.
    """
    low, high = 0.0, 20.0
    for _ in range(50):
        mid = (low + high) / 2
        if mean_margin(num_clients, noise, mid, shape) > 0:
            low = mid
        else:
            high = mid
    return high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-entropy", type=float, default=0.1, help="the project default is 0.1")
    args = parser.parse_args()

    print("F(best single-client vertex) - F(FedAvg point), mean over 5 synthetic draws.")
    print(f"Positive => the fitness optimum is degenerate. gamma_entropy = {args.gamma_entropy}.\n")
    print(f"  {'heterogeneity':>14} " + " ".join(f"{'K=' + str(k):>9}" for k in CLIENT_COUNTS))
    for noise in HETEROGENEITY:
        cells = " ".join(
            f"{mean_margin(k, noise, args.gamma_entropy):>+9.3f}" for k in CLIENT_COUNTS
        )
        print(f"  {noise:>14} {cells}")

    print("\n  Smallest gamma_entropy that keeps the corner from winning (heterogeneity 1.0):")
    for num_clients in (2, 4, 10, 20):
        needed = required_gamma_entropy(num_clients, 1.0)
        flag = "  <-- above the project default" if needed > args.gamma_entropy else ""
        print(f"    K={num_clients:<3} gamma_entropy > {needed:.3f}{flag}")

    print(
        "\n  The guard grows as log K, so a small-K validation run sits in the degenerate\n"
        "  regime while the sweep's K=20 does not -- the opposite of what a smoke test is\n"
        "  for. `corner_margin` is logged per round; read it before reading anything else."
    )

    print("\n\nThe same requirement under each penalty shape (heterogeneity 1.0).")
    print("A shape on dispersion's scale needs ONE gamma_entropy across K; log K does not.\n")
    print(f"  {'K':>4} " + " ".join(f"{shape:>10}" for shape in CONCENTRATION_PENALTIES))
    columns = {shape: [] for shape in CONCENTRATION_PENALTIES}
    for num_clients in (2, 4, 10, 20, 50):
        cells = []
        for shape in CONCENTRATION_PENALTIES:
            needed = required_gamma_entropy(num_clients, 1.0, shape)
            columns[shape].append(needed)
            cells.append(f"{needed:>10.3f}")
        print(f"  {num_clients:>4} " + " ".join(cells))
    spreads = " ".join(
        f"{max(v) / min(v):>9.2f}x" for v in (columns[s] for s in CONCENTRATION_PENALTIES)
    )
    print(f"  {'':>4} " + spreads + "   <- spread across K\n")
    print(
        "  `aco-concentration-penalty` selects the shape; \"entropy\" stays the default\n"
        "  because it is the method as proposed. The `penalty_gini` and\n"
        "  `penalty_entropy_strong` cells in configs/experiment/ablation_all.yaml measure\n"
        "  the choice on real data, which is where it should be decided."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
