"""Every number in `paper/` must still be the number its source reports.

Plan §9.3 forbids hand-typing a number into the paper, and generated tables obey that. But
`paper/04_EXPERIMENTAL_SETUP.md` is prose, and prose cannot be generated from `results/` --
the dataset section quotes measured counts that live in `data/processed/*.json` and the
manifest. Nothing connected the two, so re-running the audit with a different pHash threshold
would leave the paper asserting the old figures with no test failing.

This re-derives each quoted number from its source and asserts the paper still says it. It is
deliberately about the *dataset* numbers only: federated results belong in generated tables,
and a prose file quoting one of those is the thing this project must not start doing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "paper/04_EXPERIMENTAL_SETUP.md"


@pytest.fixture(scope="module")
def setup_text() -> str:
    return SETUP.read_text()


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data/processed/manifest.csv")


@pytest.fixture(scope="module")
def leakage() -> dict:
    return json.loads((REPO_ROOT / "data/processed/leakage_report.json").read_text())


def _says(text: str, value: str) -> bool:
    """Match a number as written, tolerating thousands separators."""
    return value in text or value.replace(",", "") in text


def test_per_class_redundancy_matches_the_manifest(setup_text, manifest) -> None:
    """The finding §4.1 turns on: no-tumour is duplicated at 2.3-2.6x the rate of every
    tumour class, which is what balancing by oversampling produces."""
    grouped = manifest.groupby("label").agg(
        images=("path", "size"), pps=("pseudo_patient_id", "nunique")
    )
    grouped["per_pp"] = grouped["images"] / grouped["pps"]

    for label, row in grouped.iterrows():
        assert _says(setup_text, f"{row['per_pp']:.3f}"), (
            f"paper does not state images-per-pseudo-patient {row['per_pp']:.3f} for {label}"
        )
        redundancy = 100 * (1 - row["pps"] / row["images"])
        assert _says(setup_text, f"{redundancy:.1f}%"), (
            f"paper does not state {redundancy:.1f}% redundancy for {label}"
        )

    # The claim is comparative, so assert the comparison rather than only the values.
    notumor = grouped.loc["notumor", "per_pp"]
    tumours = grouped.drop("notumor")["per_pp"]
    assert notumor > 2 * tumours.max(), (
        "no-tumour is no longer duplicated at more than twice the rate of every tumour class; "
        "§4.1's argument that the balance is manufactured does not hold as written"
    )


def test_dedup_class_shares_and_imbalance_ratio_match(setup_text, manifest) -> None:
    pps = manifest.groupby("label")["pseudo_patient_id"].nunique()
    for label, n in pps.items():
        share = 100 * n / pps.sum()
        assert _says(setup_text, f"{n:,}") or _says(setup_text, str(n)), \
            f"paper does not state {n} pseudo-patients for {label}"
        assert _says(setup_text, f"{share:.2f}%"), \
            f"paper does not state {share:.2f}% share for {label}"

    ratio = pps.max() / pps.min()
    assert _says(setup_text, f"{ratio:.2f}"), f"paper does not state imbalance ratio {ratio:.2f}"
    assert ratio > 2, (
        "the de-duplicated data is no longer materially imbalanced, so §4.1's macro-F1 "
        "justification -- which rests on it -- needs rewriting"
    )


def test_leakage_numbers_match_the_report(setup_text, leakage) -> None:
    totals, cross = leakage["totals"], leakage["cross_split_leakage"]
    for value in (
        totals["n_images"], totals["n_unique_sha256"], totals["n_exact_duplicate_images"],
        totals["n_pseudo_patients"], totals["n_multi_image_components"],
        totals["largest_component_size"], leakage["edges"]["n_phash_edges"],
        cross["n_leaked_components"], cross["n_leaked_images"],
        leakage["mixed_label_components"]["n"],
    ):
        assert _says(setup_text, f"{value:,}"), f"paper does not state {value:,}"

    # The headline percentage is derived, so check the derivation too.
    pct = 100 * cross["n_leaked_images"] / totals["n_images"]
    assert _says(setup_text, f"{pct:.1f}%"), f"paper does not state {pct:.1f}% leakage"


def test_split_deviation_matches(setup_text) -> None:
    stats = json.loads((REPO_ROOT / "data/processed/split_stats.json").read_text())
    assert _says(setup_text, str(stats["max_class_deviation_pp"]))
    assert stats["n_components_spanning_splits"] == 0, (
        "components span splits again; §4.2's 'zero by construction' is false"
    )


def test_the_results_section_quotes_no_federated_number(setup_text) -> None:
    """The rule this file exists to protect. A macro-F1 typed into prose diverges from the
    generated table the moment a seed is added, and nothing catches it."""
    results = (REPO_ROOT / "paper/05_RESULTS.md").read_text()
    # 0.xxxx with four decimals is what a macro-F1 looks like in this project's tables.
    suspicious = [m for m in re.findall(r"\b0\.\d{4}\b", results)]
    assert not suspicious, (
        f"05_RESULTS.md hand-types what look like metric values: {suspicious}. Generate them."
    )


def test_every_citation_placeholder_is_findable() -> None:
    """`[CITE]` markers are the handover to a human. If they are ever silently deleted rather
    than resolved, the paper ships with unsourced claims about prior work."""
    text = "".join(p.read_text() for p in sorted((REPO_ROOT / "paper").glob("*.md")))
    assert "[CITE" in text, (
        "no [CITE] markers remain in paper/. If the references were genuinely resolved, delete "
        "this test in the same commit that adds the bibliography."
    )
