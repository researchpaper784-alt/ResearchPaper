"""Phase 9, Step 9.1 -- aggregation and statistics (plan §9.1). Loads results/*.json
(the Phase 6 result contract, `utils/results.py::write_result`) into a tidy
DataFrame, computes per-(strategy, partition) summary stats across seeds, paired
Wilcoxon signed-rank tests against a reference method, Holm-Bonferroni correction
across the whole family of comparisons, Cohen's d effect sizes, and bootstrap CIs
for rounds-to-target-macro-F1.

Pure Python/pandas/scipy/numpy -- no Flower dependency, fully testable against
synthetic result-JSON fixtures matching the real schema; `scripts/aggregate_results.
py` is the thin CLI wrapper that points this at the real `results/` directory once
live runs exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def load_results(results_dir: str | Path, pattern: str = "*.json") -> pd.DataFrame:
    """One row per completed run. A run with `status != "completed"` (failed,
    unknown, or still in flight) is excluded -- it has no meaningful final metric to
    aggregate, and silently including e.g. a `None` would corrupt every mean it
    touched rather than just being absent from it."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "completed":
            continue

        config = payload.get("config", {})
        run_config = config.get("run_config", {})
        final = payload.get("final", {})
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "strategy": config.get("strategy"),
                "seed": config.get("seed"),
                "regime": run_config.get("regime"),
                "alpha": run_config.get("alpha"),
                "num_clients": run_config.get("num-clients"),
                "final_test_macro_f1": final.get("final_test_macro_f1"),
                "wall_clock_s": final.get("wall_clock_s"),
                "num_rounds_completed": final.get("num_rounds_completed"),
                "rounds": payload.get("rounds", []),
            }
        )
    return pd.DataFrame(rows)


def partition_label(row: "pd.Series[Any]") -> str:
    """A single human string identifying the partition regime + its parameter (e.g.
    'dirichlet_0.3', 'iid') so grouping by (strategy, partition) needs one column,
    not a multi-key groupby spelled out at every call site."""
    regime = row.get("regime")
    alpha = row.get("alpha")
    if regime == "dirichlet" and pd.notna(alpha):
        return f"dirichlet_{alpha}"
    return str(regime)


def with_partition_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["partition"] = df.apply(partition_label, axis=1)
    return df


def summarize(df: pd.DataFrame, metric: str = "final_test_macro_f1") -> pd.DataFrame:
    """Mean +/- std of `metric`, grouped by (strategy, partition), across seeds."""
    return (
        with_partition_column(df)
        .groupby(["strategy", "partition"])[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def paired_values(
    df: pd.DataFrame, strategy_a: str, strategy_b: str, partition: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Values for `strategy_a`/`strategy_b` at `partition`, paired by seed -- a seed
    present for only one of the two strategies contributes nothing to a paired test
    and is dropped, not treated as a missing 0."""
    df = with_partition_column(df)
    a = df[(df["strategy"] == strategy_a) & (df["partition"] == partition)].set_index("seed")[metric]
    b = df[(df["strategy"] == strategy_b) & (df["partition"] == partition)].set_index("seed")[metric]
    common_seeds = a.index.intersection(b.index)
    return a.loc[common_seeds].to_numpy(dtype=float), b.loc[common_seeds].to_numpy(dtype=float)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standard pooled-variance Cohen's d. Returns 0.0 (not nan/inf) when both
    samples have zero variance and equal means -- a genuine "no effect," distinct
    from the zero-variance-but-different-means case, which is mathematically an
    infinite effect size and IS left as inf rather than silently clamped."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0 if a.mean() == b.mean() else float("inf") * np.sign(a.mean() - b.mean())
    return float((a.mean() - b.mean()) / pooled_std)


def wilcoxon_paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired (a, b) (plan §9.1: "do not assume
    normality with n=5"). scipy raises ValueError when every paired difference is
    zero or too few nonzero differences exist to compute a statistic -- caught and
    reported as nan rather than propagated, since "no detectable difference" is a
    real, reportable outcome here, not a crash."""
    n_pairs = len(a)
    if n_pairs < 1:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": 0}
    diffs = a - b
    if np.all(diffs == 0):
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": n_pairs}
    try:
        result = stats.wilcoxon(a, b)
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_pairs": n_pairs}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": n_pairs}


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction (plan §9.1: correct across the whole
    family of comparisons, not per-partition separately -- doing the latter would
    under-correct). Returns adjusted p-values in the SAME order as `p_values`. A nan
    input p-value (an incomputable test, not a real result) stays nan in the output
    and does not participate in the step-down ordering of the others."""
    m = len(p_values)
    p_array = np.array(p_values, dtype=float)
    valid_mask = ~np.isnan(p_array)
    adjusted = np.full(m, np.nan)

    valid_p = p_array[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return adjusted.tolist()

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    adjusted_sorted = np.empty(n_valid)
    running_max = 0.0
    for i in range(n_valid):
        candidate = min((n_valid - i) * sorted_p[i], 1.0)
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = running_max
    adjusted_valid = np.empty(n_valid)
    adjusted_valid[order] = adjusted_sorted
    adjusted[valid_mask] = adjusted_valid
    return adjusted.tolist()


def compare_against_baselines(
    df: pd.DataFrame,
    method: str,
    baselines: list[str],
    partitions: list[str],
    metric: str = "final_test_macro_f1",
) -> pd.DataFrame:
    """Every (baseline, partition) comparison against `method`: paired Wilcoxon +
    Cohen's d per cell, Holm-Bonferroni correction applied ONCE across every row of
    the returned table (the whole family), not per partition. Empty (no overlapping
    seeds) cells are skipped, not included as all-nan rows."""
    rows = []
    for partition in partitions:
        for baseline in baselines:
            a, b = paired_values(df, method, baseline, partition, metric)
            if len(a) == 0:
                continue
            test = wilcoxon_paired_test(a, b)
            rows.append(
                {
                    "partition": partition,
                    "baseline": baseline,
                    "method_mean": float(a.mean()),
                    "baseline_mean": float(b.mean()),
                    "cohens_d": cohens_d(a, b),
                    **test,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = holm_bonferroni(result["p_value"].tolist())
    return result


def rounds_to_target(rounds: list[dict[str, Any]], target: float, metric_key: str = "test_macro_f1") -> int:
    """First round number at which `metric_key >= target`. A run that never reaches
    it returns its last logged round number as a censored lower bound, not `None`
    -- silently dropping never-reached seeds from an aggregate would bias the
    estimate optimistic by keeping only the seeds that succeeded."""
    for entry in rounds:
        if entry.get(metric_key, float("-inf")) >= target:
            return int(entry["round"])
    return int(rounds[-1]["round"]) if rounds else 0


def bootstrap_rounds_to_target(
    rounds_log_per_seed: list[list[dict[str, Any]]],
    target: float,
    metric_key: str = "test_macro_f1",
    num_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: "np.random.Generator | None" = None,
) -> dict[str, float]:
    """Bootstrap CI (plan §9.1) for rounds-to-target-macro-F1, resampling across
    seeds (each element of `rounds_log_per_seed` is one seed's full `rounds` log)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    per_seed = np.array([rounds_to_target(r, target, metric_key) for r in rounds_log_per_seed], dtype=float)
    if len(per_seed) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    resampled_means = np.array(
        [rng.choice(per_seed, size=len(per_seed), replace=True).mean() for _ in range(num_bootstrap)]
    )
    alpha = 1.0 - ci
    return {
        "mean": float(per_seed.mean()),
        "ci_low": float(np.quantile(resampled_means, alpha / 2)),
        "ci_high": float(np.quantile(resampled_means, 1 - alpha / 2)),
    }
