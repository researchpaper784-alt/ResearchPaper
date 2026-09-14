"""Phase 1.4 — partition diagnostic figures.

One figure per regime: a clients x classes heatmap alongside the client-size distribution,
annotated with the two heterogeneity scalars. Plus a summary comparing regimes on both
axes, which is the figure that shows label skew and size skew are independent.

Run: make figures-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedswarm.data.datasets import load_manifest  # noqa: E402
from fedswarm.data.partition import PartitionSpec, load_or_build  # noqa: E402

FIGURE_DIR = Path("paper/figures")

REGIMES: dict[str, PartitionSpec] = {
    "iid": PartitionSpec(regime="iid"),
    "dirichlet_0.1": PartitionSpec(regime="dirichlet", alpha=0.1),
    "dirichlet_0.3": PartitionSpec(regime="dirichlet", alpha=0.3),
    "dirichlet_0.5": PartitionSpec(regime="dirichlet", alpha=0.5),
    "dirichlet_1.0": PartitionSpec(regime="dirichlet", alpha=1.0),
    "pathological_2": PartitionSpec(regime="pathological", classes_per_client=2),
    "quantity_skew": PartitionSpec(regime="quantity_skew", skew_sigma=1.0),
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def regime_figure(name: str, diagnostics: dict, out: Path) -> None:
    matrix = np.array(diagnostics["class_matrix"])
    sizes = np.array(diagnostics["client_sizes"])
    classes = diagnostics["classes"]

    fig, (heat_ax, hist_ax) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # Row-normalize so the heatmap shows composition, not client size.
    composition = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image = heat_ax.imshow(composition, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    heat_ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    heat_ax.set_yticks(range(0, len(matrix), max(1, len(matrix) // 10)))
    heat_ax.set_ylabel("client")
    heat_ax.set_title(f"{name}\nclass composition per client")
    fig.colorbar(image, ax=heat_ax, label="fraction of client's data")

    hist_ax.bar(range(len(sizes)), np.sort(sizes)[::-1], color="#4C72B0")
    hist_ax.set_xlabel("client (sorted by size)")
    hist_ax.set_ylabel("pseudo-patients")
    hist_ax.set_title(
        f"JS divergence {diagnostics['js_divergence']:.3f}   "
        f"size Gini {diagnostics['size_gini']:.3f}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def summary_figure(collected: dict[str, dict], out: Path) -> None:
    """Label skew against size skew. Regimes that separate on these axes are measuring
    genuinely different kinds of heterogeneity, which is what lets the paper attribute a
    gain to one or the other."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for name, diagnostics in collected.items():
        ax.scatter(
            diagnostics["js_divergence"],
            diagnostics["size_gini"],
            s=70,
            label=name,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            name,
            (diagnostics["js_divergence"], diagnostics["size_gini"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("label skew  (mean pairwise Jensen-Shannon divergence)")
    ax.set_ylabel("size skew  (Gini coefficient of client sizes)")
    ax.set_title("Partition regimes separate label skew from size skew")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(FIGURE_DIR))
    args = parser.parse_args()

    style()
    manifest = load_manifest()
    out_dir = Path(args.out_dir)
    collected = {}

    for name, base in REGIMES.items():
        spec = PartitionSpec(
            regime=base.regime,
            num_clients=args.num_clients,
            seed=args.seed,
            alpha=base.alpha,
            classes_per_client=base.classes_per_client,
            skew_sigma=base.skew_sigma,
        )
        diagnostics = load_or_build(manifest, spec).diagnostics
        collected[name] = diagnostics
        regime_figure(name, diagnostics, out_dir / f"partition_{name}.pdf")
        print(
            f"{name:<16} JS={diagnostics['js_divergence']:.3f}  "
            f"gini={diagnostics['size_gini']:.3f}  "
            f"min={min(diagnostics['client_sizes'])}  "
            f"max={max(diagnostics['client_sizes'])}"
        )

    summary_figure(collected, out_dir / "partition_summary.pdf")
    print(f"\nWrote {len(collected) + 1} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
