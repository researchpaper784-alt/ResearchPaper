"""Phase 3.2 tests -- the parts of fl/server_app.py that don't need a real Flower
`Grid`: `global_test_loader` (plain data loading) and `build_evaluate_fn`'s returned
closure (a plain `Callable[[int, ArrayRecord], MetricRecord]` -- `Strategy.start()`
calls it, but nothing stops calling it directly the same way here).

`@app.main()` itself is not called in any test: it needs a real `Grid`, which needs
the `simulation` extra this machine cannot install (`docs/FLOWER_API_NOTES.md`). The
first real `flwr run` on Colab is what actually exercises it end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from flwr.app import ArrayRecord
from PIL import Image

from fedswarm.data.cache import build_and_save_cache
from fedswarm.fl.checkpoint import load_checkpoint
from fedswarm.fl.server_app import build_evaluate_fn, global_test_loader
from fedswarm.models.simple_cnn import SimpleCNN


@pytest.fixture
def run_config_with_test_split(tmp_path: Path) -> dict:
    root = tmp_path / "raw"
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)

    rows = []
    for i in range(20):
        split_dir, split = ("Training", "train") if i < 12 else ("Testing", "test")
        rel = f"{split_dir}/glioma/img_{i}.jpg"
        Image.new("L", (16, 16), color=(i * 11) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if i % 2 == 0 else "notumor",
                "pseudo_patient_id": i,
                "split": split,
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = tmp_path / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)

    return {
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "model-name": "simple_cnn",
        "model-norm": "groupnorm",
        "num-classes": 4,
        "checkpoint-dir": str(tmp_path / "checkpoints"),
        "_run_id": "test-run",
    }


def test_global_test_loader_only_contains_test_split_rows(run_config_with_test_split) -> None:
    loader = global_test_loader(run_config_with_test_split)
    assert len(loader.dataset) == 8  # rows 12..19 above


def test_evaluate_fn_returns_metric_record_and_logs_a_round(run_config_with_test_split) -> None:
    rounds_log: list[dict] = []
    evaluate_fn = build_evaluate_fn(run_config_with_test_split, rounds_log, round_offset=0)

    model = SimpleCNN(num_classes=4, norm="groupnorm")
    result = evaluate_fn(1, ArrayRecord(model.state_dict()))

    assert "test_macro_f1" in dict(result)
    assert len(rounds_log) == 1
    assert rounds_log[0]["round"] == 1
    assert "test_macro_f1" in rounds_log[0]


def test_evaluate_fn_checkpoints_after_every_round(run_config_with_test_split) -> None:
    rounds_log: list[dict] = []
    evaluate_fn = build_evaluate_fn(run_config_with_test_split, rounds_log, round_offset=0)
    model = SimpleCNN(num_classes=4, norm="groupnorm")

    evaluate_fn(1, ArrayRecord(model.state_dict()))

    checkpoint = load_checkpoint(run_config_with_test_split["checkpoint-dir"], "test-run")
    assert checkpoint is not None
    round_number, logged, _state = checkpoint
    assert round_number == 1
    assert logged == rounds_log


def test_evaluate_fn_skips_logging_the_resumed_round_zero_reconfirmation(run_config_with_test_split) -> None:
    """On a resumed run, Strategy.start()'s own pre-round-1 call re-evaluates the exact
    checkpointed weights already logged by the previous invocation as round_offset --
    round_offset > 0 means we're mid-resume, and server_round == 0 is that duplicate."""
    rounds_log: list[dict] = [{"round": 5, "test_macro_f1": 0.7}]  # pretend round 5 already happened
    evaluate_fn = build_evaluate_fn(run_config_with_test_split, rounds_log, round_offset=5)
    model = SimpleCNN(num_classes=4, norm="groupnorm")

    evaluate_fn(0, ArrayRecord(model.state_dict()))  # this run's internal round 0 == true round 5

    assert len(rounds_log) == 1  # unchanged: not appended again

    evaluate_fn(1, ArrayRecord(model.state_dict()))  # true round 6: a genuinely new round

    assert len(rounds_log) == 2
    assert rounds_log[1]["round"] == 6
