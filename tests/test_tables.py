"""Phase 9 tests -- scripts/make_tables.py.

A results table misleads by omission far more easily than by arithmetic. These test the
omissions: a cell quietly averaged over fewer seeds than its neighbours, two different
experimental conditions pooled into one row, and a FedACO score presented without the
two numbers that say whether its mechanism did anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_tables import (  # noqa: E402
    to_latex,
    add_significance,
    power_note,
    _variant_of,
    add_deltas,
    cell_key,
    summarize,
    to_markdown,
)


def _result(strategy, regime, seed, test_f1, val_f1=None, rounds=None, **run_config):
    config = {"regime": regime, "seed": seed, "strategy-name": strategy, **run_config}
    return {
        "status": "completed",
        "config": {"strategy": strategy, "seed": seed, "run_config": config},
        "rounds": rounds or [],
        "final": {"final_test_macro_f1": test_f1, "best_val_macro_f1": val_f1},
    }


# ======================================================================================
# Grouping
# ======================================================================================


def test_dirichlet_alphas_are_never_pooled() -> None:
    """dirichlet@0.1 and dirichlet@1.0 are different experimental conditions -- Phase 1.4
    measured JS divergence 0.541 vs 0.217. Pooling them into one `dirichlet` row would
    average across a factor-of-two difference in heterogeneity."""
    a = cell_key(_result("fedavg", "dirichlet", 0, 0.8, alpha=0.1))
    b = cell_key(_result("fedavg", "dirichlet", 0, 0.8, alpha=1.0))

    assert a != b
    assert a[1] == "dirichlet_0.1" and b[1] == "dirichlet_1.0"


def test_variant_is_recovered_from_the_resolved_config() -> None:
    """A result file records the resolved config, not the sweep's variant name, so the
    ablation has to be reconstructed from the knobs that differ from default."""
    assert _variant_of({}) == "default"
    assert "persistence=none" in _variant_of({"aco-persistence": "none"})
    assert "norm=batchnorm" in _variant_of({"model-norm": "batchnorm"})
    assert "no-fallback" in _variant_of({"aco-safety-fallback": False})
    assert "s=0.8" in _variant_of({"aco-target-sum": 0.8})


def test_ablation_variants_do_not_pool_with_the_control() -> None:
    control = cell_key(_result("fedaco", "iid", 0, 0.8))
    ablated = cell_key(_result("fedaco", "iid", 0, 0.8, **{"aco-persistence": "none"}))

    assert control != ablated


# ======================================================================================
# Incompleteness
# ======================================================================================


def test_a_cell_with_missing_seeds_is_marked_not_dropped() -> None:
    """Silently averaging whatever finished is how one strategy's mean over 5 seeds ends
    up next to another's over 2, presented identically."""
    results = [_result("fedavg", "iid", s, 0.9) for s in (0, 1)]

    rows = summarize(results, expected_seeds=5)

    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["incomplete"] is True


def test_a_complete_cell_is_not_marked() -> None:
    results = [_result("fedavg", "iid", s, 0.9) for s in range(5)]

    rows = summarize(results, expected_seeds=5)

    assert rows[0]["n"] == 5 and rows[0]["incomplete"] is False


def test_markdown_warns_about_incomplete_cells() -> None:
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=5)
    add_deltas(rows, "fedavg")

    markdown = to_markdown(rows, "fedavg")

    assert "fewer seeds than expected" in markdown
    assert "⚠️" in markdown


# ======================================================================================
# Deltas
# ======================================================================================


def test_delta_is_paired_by_seed() -> None:
    """Both methods saw the same seeds, so the per-seed difference has far less variance
    than the difference of the means."""
    results = [_result("fedavg", "iid", s, 0.80 + 0.01 * s) for s in range(3)]
    results += [_result("fedaco", "iid", s, 0.85 + 0.01 * s) for s in range(3)]

    rows = summarize(results, expected_seeds=3)
    add_deltas(rows, "fedavg")

    aco = next(r for r in rows if r["strategy"] == "fedaco")
    assert aco["delta_paired"] is True
    assert abs(aco["delta"] - 0.05) < 1e-9


def test_delta_is_none_when_no_shared_seeds_exist() -> None:
    """Better an explicit gap than a mean-to-mean number quietly presented as if it were
    the paired comparison."""
    results = [_result("fedavg", "iid", 0, 0.80), _result("fedaco", "iid", 7, 0.85)]

    rows = summarize(results, expected_seeds=1)
    add_deltas(rows, "fedavg")

    aco = next(r for r in rows if r["strategy"] == "fedaco")
    assert aco["delta"] is None and aco["delta_paired"] is False


def test_baseline_row_has_zero_delta() -> None:
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    assert rows[0]["delta"] == 0.0


# ======================================================================================
# FedACO health columns
# ======================================================================================


def test_fedaco_rows_carry_fallback_and_entropy() -> None:
    """A headline macro-F1 says nothing about whether the colony searched or whether the
    fallback quietly turned FedACO into FedAvg. Both must sit next to the score."""
    rounds = [
        {"round": 1, "train_fallback_used": 1, "train_pheromone_entropy": 2.39},
        {"round": 2, "train_fallback_used": 0, "train_pheromone_entropy": 2.37},
    ]
    rows = summarize([_result("fedaco", "iid", 0, 0.9, rounds=rounds)], expected_seeds=1)

    assert rows[0]["fallback_rate"] == 0.5
    assert abs(rows[0]["pheromone_entropy"] - 2.38) < 1e-9


def test_markdown_explains_how_to_read_the_health_columns() -> None:
    rounds = [{"round": 1, "train_fallback_used": 1, "train_pheromone_entropy": 2.39}]
    rows = summarize([_result("fedaco", "iid", 0, 0.9, rounds=rounds)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    markdown = to_markdown(rows, "fedavg")

    assert "silently *was* FedAvg" in markdown
    assert "not searching" in markdown


def test_rows_without_health_metrics_render_as_dashes_not_zeros() -> None:
    """A baseline has no pheromone. Rendering that as 0.000 would read as 'the colony was
    maximally concentrated', the opposite of 'not applicable'."""
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    assert rows[0]["pheromone_entropy"] is None
    assert "| — | — |" in to_markdown(rows, "fedavg")


# ======================================================================================
# Significance, and whether the table can support the claim at all
# ======================================================================================


def _best_case(n_seeds: int, regimes=("iid",)):
    """FedACO ahead of FedAvg on every seed in every regime -- the most favourable data
    that can exist. Any "not significant" here is about the sample size."""
    results = []
    for regime in regimes:
        for seed in range(n_seeds):
            results.append(_result("fedavg", regime, seed, 0.880 + 0.002 * seed))
            results.append(_result("fedaco", regime, seed, 0.905 + 0.002 * seed))
    return results


def _rows(results, expected_seeds, alternative="two-sided"):
    rows = summarize(results, expected_seeds)
    add_deltas(rows, "fedavg")
    add_significance(rows, "fedavg", alternative)
    return rows


def test_the_table_reports_a_test_not_just_a_margin() -> None:
    """Mean +/- std invites a reader to eyeball two overlapping error bars and conclude
    whatever they arrived believing. At n=5 the seed spread is comparable to the margins
    these methods differ by, which is exactly when that goes wrong."""
    rows = _rows(_best_case(5), 5)
    aco = next(r for r in rows if r["strategy"] == "fedaco")

    assert aco["p_value"] is not None
    assert aco["p_value_holm"] is not None
    assert aco["cohens_d"] > 0


def test_the_baseline_is_not_tested_against_itself() -> None:
    rows = _rows(_best_case(5), 5)
    fedavg = next(r for r in rows if r["strategy"] == "fedavg")

    assert fedavg["p_value"] is None and fedavg["cohens_d"] is None


def test_holm_corrects_across_the_whole_table_not_per_regime() -> None:
    """Correcting inside each regime and then reporting six regimes under-corrects by
    exactly the factor the correction exists to supply."""
    one_regime = _rows(_best_case(8, ("iid",)), 8)
    six_regimes = _rows(_best_case(8, tuple(f"r{i}" for i in range(6))), 8)

    single = next(r for r in one_regime if r["strategy"] == "fedaco")["p_value_holm"]
    family = next(r for r in six_regimes if r["strategy"] == "fedaco")["p_value_holm"]

    assert family > single, "a larger family must not get the same adjusted p"


def test_a_table_that_cannot_reach_alpha_says_so() -> None:
    """The check that makes the statistics honest rather than decorative.

    At 5 seeds the signed-rank test's smallest possible two-sided p is 0.0625, so the
    main sweep's planned seed count cannot produce a significant result at alpha=0.05 --
    with Cohen's d near 8 and every seed favouring the method. Reporting "not
    significant" there without saying why would be the single most misleading number the
    paper could contain.
    """
    rows = _rows(_best_case(5), 5)
    note = power_note(rows, alpha=0.05, alternative="two-sided")

    assert note is not None
    assert "cannot reach" in note
    assert "6 seeds" in note, "it has to say what would be enough for this family"

    # And the requirement scales with the family -- asserted as the relationship rather
    # than a hard-coded count, because the count depends on how many claims the table
    # makes and is exactly the thing that is easy to miscount by hand.
    from fedswarm.analysis import seeds_needed_for

    assert seeds_needed_for(0.05, 6) > seeds_needed_for(0.05, 1)
    assert seeds_needed_for(0.05, 66) > seeds_needed_for(0.05, 6)
    # And the effect it is failing to detect is enormous, which is the point.
    assert next(r for r in rows if r["strategy"] == "fedaco")["cohens_d"] > 3


def test_the_warning_is_silent_when_the_test_can_actually_fire() -> None:
    """The other half: a check that always warns is noise, not a safeguard."""
    rows = _rows(_best_case(8), 8)

    assert power_note(rows, alpha=0.05, alternative="two-sided") is None
    assert next(r for r in rows if r["strategy"] == "fedaco")["p_value_holm"] < 0.05


def test_a_one_sided_test_is_available_but_not_the_default() -> None:
    """Halving the achievable p is legitimate for a directional hypothesis and illegitimate
    if chosen after seeing the two-sided result, so it must be opt-in."""
    two_sided = _rows(_best_case(6), 6, "two-sided")
    one_sided = _rows(_best_case(6), 6, "greater")

    p_two = next(r for r in two_sided if r["strategy"] == "fedaco")["p_value"]
    p_one = next(r for r in one_sided if r["strategy"] == "fedaco")["p_value"]
    assert p_one < p_two


# ======================================================================================
# Phase 9.3 -- LaTeX output
# ======================================================================================


def test_the_latex_table_is_booktabs_with_the_best_in_bold(tmp_path: Path) -> None:
    """Plan §9.3's deliverable, and its instruction: "never hand-type a number into the
    paper". The .tex comes from the same rows as the .md and .csv, so all three cannot
    disagree."""
    rows = _rows(_best_case(8), 8)
    out = tmp_path / "main.tex"

    to_latex(rows, "fedavg", out, "main")
    tex = out.read_text()

    for required in ("\\toprule", "\\midrule", "\\bottomrule", "\\begin{tabular}"):
        assert required in tex, required
    assert "\\textbf{" in tex, "the best cell per partition must be bold"
    assert "\\label{tab:main}" in tex


def test_significance_markers_appear_only_when_significant(tmp_path: Path) -> None:
    """At 8 seeds the best case reaches p=0.008 and earns `**`; at 5 seeds the same data
    floors at 0.0625 and must earn nothing. A table that marked both would be asserting
    significance the test never found."""
    eight = tmp_path / "eight.tex"
    five = tmp_path / "five.tex"

    to_latex(_rows(_best_case(8), 8), "fedavg", eight, "eight")
    to_latex(_rows(_best_case(5), 5), "fedavg", five, "five")

    assert "**" in eight.read_text()
    assert "*" not in five.read_text().split("\\midrule")[1], "no marker is achievable at n=5"


def test_regenerating_the_table_is_byte_for_byte_identical(tmp_path: Path) -> None:
    """Plan §9.3's literal acceptance criterion: "Deleting paper/figures/ and
    paper/tables/ and re-running restores them byte-for-byte identical." A timestamp, a
    dict iteration order or a set ordering anywhere in the path would break it, and the
    breakage would only show up as noise in the paper's diff."""
    rows = _rows(_best_case(8), 8)
    first, second = tmp_path / "a.tex", tmp_path / "b.tex"

    to_latex(rows, "fedavg", first, "main")
    to_latex(_rows(_best_case(8), 8), "fedavg", second, "main")

    assert first.read_bytes() == second.read_bytes()


def test_an_ablation_table_is_labelled_by_variant_not_strategy(tmp_path: Path) -> None:
    """Every row of an ablation sweep is FedACO. Labelling by strategy would print a
    column of identical "fedaco" rows and lose the thing being ablated."""
    results = []
    for fallback_off, score in ((False, 0.90), (True, 0.87)):
        for seed in range(5):
            # dirichlet_0.3 carries an underscore, which LaTeX needs escaped -- the
            # variant labels this sweep produces happen to be hyphenated.
            r = _result("fedaco", "dirichlet", seed, score + 0.001 * seed, alpha=0.3)
            if fallback_off:
                r["config"]["run_config"]["aco-safety-fallback"] = False
            results.append(r)
    rows = summarize(results, 5)
    add_deltas(rows, "fedavg")
    add_significance(rows, "fedavg")
    out = tmp_path / "abl.tex"

    to_latex(rows, "fedavg", out, "ablation")
    tex = out.read_text()

    assert {r["variant"] for r in rows} == {"default", "no-fallback"}
    assert "default" in tex and "no-fallback" in tex
    assert tex.count("fedaco") == 0, "ablation rows are the variants, not the strategy"
    assert "dirichlet\\_0.3" in tex, "underscores must be escaped for LaTeX"


# ======================================================================================
# The LaTeX table is what goes in the paper, and it was dropping both warnings
# ======================================================================================

def _latex_rows(fedaco_n: int, fedavg_n: int = 8, expected: int = 8) -> list[dict]:
    """Real rows via `summarize`, not hand-built dicts: `add_significance` reads `by_seed`,
    which only the real pipeline produces, and a fixture that skips it would test a shape the
    code never sees."""
    results = []
    for seed in range(fedavg_n):
        results.append(_result("fedavg", "iid", seed, 0.880 + 0.002 * seed))
    for seed in range(fedaco_n):
        results.append(_result("fedaco", "iid", seed, 0.905 + 0.002 * seed))
    return _rows(results, expected)


def test_an_incomplete_cell_is_marked_in_the_latex_not_only_the_markdown(tmp_path: Path) -> None:
    """The defect: `to_latex` built its frame from strategy/partition/mean/std/n and dropped
    `expected_n`, so a cell averaging 3 of 8 seeds printed `0.510 $\\pm$ 0.010` --
    character-for-character identical to a complete one -- while the markdown beside it showed
    `3 ⚠️` and a paragraph of warning.

    The markdown is read by whoever ran the sweep. The LaTeX is what goes in the paper, and
    the main sweep runs over many sessions, so a partially-filled table is its normal state
    for weeks. The plan's own rule is "never hand-type a number into the paper", which makes
    this emitter the number.
    """
    path = tmp_path / "t.tex"
    to_latex(_latex_rows(fedaco_n=3), "fedavg", path, "t")
    tex = path.read_text()

    assert "$^{\\dagger}$" in tex, "the incomplete cell carries no marker"
    # In the caption, because a caption is the only thing that travels with a table someone
    # pastes into a draft.
    assert "fewer seeds than planned" in tex
    assert "fedaco/iid (3 of 8)" in tex
    assert "Do not quote these numbers" in tex


def test_a_complete_table_carries_no_dagger(tmp_path: Path) -> None:
    """The warning has to be absent when it does not apply, or it becomes wallpaper."""
    path = tmp_path / "t.tex"
    to_latex(_latex_rows(fedaco_n=8), "fedavg", path, "t")

    assert "\\dagger" not in path.read_text()


def test_the_significance_legend_is_withheld_when_the_alpha_is_unreachable(tmp_path: Path) -> None:
    """A caption printing `$^{*}p<0.05$` while the signed-rank test cannot reach 0.05 at this
    pair count advertises a threshold no data can cross, and a reader of the paper has no way
    to know. The markdown said so in a paragraph the LaTeX never carried."""
    from make_tables import _latex_power_note

    rows = _latex_rows(fedaco_n=3, fedavg_n=3, expected=3)
    warning = _latex_power_note(rows, alpha=0.05, alternative="two-sided")
    assert warning is not None, "n=3 pairs cannot reach 0.05; the note should fire"

    path = tmp_path / "t.tex"
    to_latex(rows, "fedavg", path, "t", power_warning=warning)
    tex = path.read_text()

    assert "$^{*}p<0.05$" not in tex, "the legend promises a threshold the test cannot cross"
    assert "unreachable at any data" in tex
    assert "reflects the sample size rather than the effect" in tex


def test_an_adequately_powered_table_keeps_its_legend(tmp_path: Path) -> None:
    from make_tables import _latex_power_note

    rows = _latex_rows(fedaco_n=8, fedavg_n=8, expected=8)

    assert _latex_power_note(rows, alpha=0.05, alternative="two-sided") is None

    path = tmp_path / "t.tex"
    to_latex(rows, "fedavg", path, "t", power_warning=None)

    assert "$^{*}p<0.05$" in path.read_text()


def test_both_power_notes_agree_about_whether_a_table_is_underpowered() -> None:
    """The markdown and LaTeX warnings compute the same thing, and two formatters computing it
    separately is exactly how the markdown ended up warning about something the LaTeX
    advertised. They share `_power_floor`; this pins that they cannot diverge."""
    from make_tables import _latex_power_note

    for n in (3, 5, 6, 8, 10):
        rows = _latex_rows(fedaco_n=n, fedavg_n=n, expected=n)
        markdown = power_note(rows, alpha=0.05, alternative="two-sided")
        latex = _latex_power_note(rows, alpha=0.05, alternative="two-sided")

        assert (markdown is None) == (latex is None), (
            f"at n={n} the markdown and LaTeX disagree about whether this table is "
            f"underpowered: markdown={markdown is not None}, latex={latex is not None}"
        )


# ======================================================================================
# The variant label must be derived from the real defaults, never from a literal
# ======================================================================================


def test_a_run_at_the_shipped_defaults_is_labelled_default() -> None:
    """`_variant_of` used to hardcode the default it compared each knob against. On
    2026-09-24 `aco-gamma-dispersion` was reconciled to plan §14's 0.50 (it had drifted to
    1.0 in pyproject) and every FedACO run promptly read as a `g2=0.5` variant.

    That is not cosmetic: `add_deltas` computes a delta only for rows whose variant is
    "default", so every comparison column in every table would have come out empty -- from a
    correct change to a value, made in the right place.
    """
    import tomllib
    from pathlib import Path

    import make_tables

    defaults = tomllib.loads(
        (Path(make_tables.__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["tool"]["flwr"]["app"]["config"]

    make_tables._defaults.cache_clear()
    assert make_tables._variant_of(dict(defaults)) == "default"


def test_each_knob_off_default_is_named() -> None:
    import tomllib
    from pathlib import Path

    import make_tables

    root = Path(make_tables.__file__).resolve().parents[1]
    defaults = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    make_tables._defaults.cache_clear()

    for key, value, expected in [
        ("aco-persistence", "none", "persistence=none"),
        ("aco-fitness-mode", "server_val", "fitness=server_val"),
        ("model-norm", "batchnorm", "norm=batchnorm"),
        ("aco-q0", 0.9, "q0=0.9"),
        ("aco-desirability-scaling", "standardized", "scaling=standardized"),
        ("aco-dispersion-reference", "aggregate", "ref=aggregate"),
    ]:
        assert defaults[key] != value, f"{key} test value equals the default; pick another"
        assert make_tables._variant_of({**defaults, key: value}) == expected


def test_the_variant_label_tracks_a_default_that_moves() -> None:
    """The property that failed: change a default and a run *at* that new default must still
    be "default", while a run at the old value becomes the variant."""
    import make_tables

    real = make_tables._defaults
    try:
        make_tables._defaults = lambda: {"aco-q0": 0.5}
        assert make_tables._variant_of({"aco-q0": 0.5}) == "default"
        assert make_tables._variant_of({"aco-q0": 0.7}) == "q0=0.7"
    finally:
        make_tables._defaults = real


def test_an_unknown_key_in_the_config_does_not_become_a_variant() -> None:
    """A key absent from pyproject cannot be compared, and guessing would turn any new
    run_config key into a spurious variant across every existing result."""
    import make_tables

    make_tables._defaults.cache_clear()
    assert make_tables._variant_of({"some-future-key": 42}) == "default"
