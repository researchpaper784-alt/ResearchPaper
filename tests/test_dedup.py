"""Phase 1.2 tests — verified against synthetic fixtures with known ground truth, so the
de-duplication logic is falsifiable without needing the real dataset on disk."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fedswarm.data.dedup import (
    audit,
    components_from_edges,
    exact_duplicate_edges,
    near_duplicate_edges,
)


def _write_noise_image(path: Path, seed: int, size: int = 128) -> None:
    """A distinctive image. Low-frequency blobs, not white noise -- phash works on the
    DCT's low-frequency corner, and pure noise images hash near-identically."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    img = Image.fromarray(small, mode="L").resize((size, size), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def _brighten(src: Path, dst: Path, delta: int = 6) -> None:
    """A near-duplicate: same structure, slightly different intensity -- the stand-in for
    consecutive slices of one patient volume."""
    with Image.open(src) as img:
        arr = np.asarray(img.convert("L")).astype(np.int16)
    arr = np.clip(arr + delta, 0, 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(dst, quality=95)


@pytest.fixture
def planted_dataset(tmp_path: Path) -> Path:
    """A miniature dataset with planted, known-by-construction duplicates:

      * img_000 in Training/glioma is byte-copied to Training/glioma/exact_copy.jpg
      * img_001 in Training/glioma has a brightened near-duplicate in Testing/glioma
        -> this is a CROSS-SPLIT LEAK and the audit must find it
      * everything else is distinct
    """
    root = tmp_path / "brain-tumor-mri"

    for split in ("Training", "Testing"):
        for cls in ("glioma", "meningioma", "notumor", "pituitary"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)

    for i in range(6):
        _write_noise_image(root / "Training" / "glioma" / f"img_{i:03d}.jpg", seed=100 + i)
    for i in range(4):
        _write_noise_image(root / "Training" / "meningioma" / f"img_{i:03d}.jpg", seed=200 + i)
    for i in range(3):
        _write_noise_image(root / "Testing" / "notumor" / f"img_{i:03d}.jpg", seed=300 + i)

    # Planted exact duplicate (same bytes, different filename, same split).
    shutil.copyfile(
        root / "Training" / "glioma" / "img_000.jpg",
        root / "Training" / "glioma" / "exact_copy.jpg",
    )

    # Planted cross-split near-duplicate.
    _brighten(
        root / "Training" / "glioma" / "img_001.jpg",
        root / "Testing" / "glioma" / "near_dup_of_train_001.jpg",
    )

    return root


def test_exact_duplicate_edges_finds_byte_identical_files() -> None:
    shas = ["aaa", "bbb", "aaa", "ccc", "bbb", "bbb"]
    edges = exact_duplicate_edges(shas)
    labels = components_from_edges(len(shas), edges)

    assert labels[0] == labels[2]
    assert labels[1] == labels[4] == labels[5]
    assert labels[3] not in {labels[0], labels[1]}


def test_near_duplicate_edges_matches_bruteforce_hamming() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(40, 64)).astype(np.float32)

    edges = set(near_duplicate_edges(bits, threshold=12, chunk_size=7))

    expected = {
        (i, j)
        for i in range(40)
        for j in range(i + 1, 40)
        if int(np.sum(bits[i] != bits[j])) <= 12
    }
    assert edges == expected


def test_near_duplicate_edges_excludes_self_pairs() -> None:
    bits = np.zeros((5, 64), dtype=np.float32)
    edges = near_duplicate_edges(bits, threshold=0)

    assert all(i < j for i, j in edges)
    assert len(edges) == 10  # all 5 identical -> every upper-triangle pair


def test_audit_detects_planted_exact_duplicate(planted_dataset: Path) -> None:
    result = audit(planted_dataset, phash_threshold=5, show_progress=False)
    report, records = result["report"], result["records"]

    assert report["totals"]["n_exact_duplicate_images"] == 1

    by_path = {r["path"]: r for r in records}
    original = by_path["Training/glioma/img_000.jpg"]
    copy = by_path["Training/glioma/exact_copy.jpg"]
    assert original["pseudo_patient_id"] == copy["pseudo_patient_id"]


def test_audit_detects_planted_cross_split_leak(planted_dataset: Path) -> None:
    result = audit(planted_dataset, phash_threshold=5, show_progress=False)
    report, records = result["report"], result["records"]

    assert report["cross_split_leakage"]["n_leaked_components"] >= 1

    by_path = {r["path"]: r for r in records}
    train_img = by_path["Training/glioma/img_001.jpg"]
    test_img = by_path["Testing/glioma/near_dup_of_train_001.jpg"]
    assert train_img["pseudo_patient_id"] == test_img["pseudo_patient_id"], (
        "the brightened near-duplicate must land in the same pseudo-patient as its source"
    )


def test_audit_keeps_distinct_images_separate(planted_dataset: Path) -> None:
    result = audit(planted_dataset, phash_threshold=5, show_progress=False)
    records = result["records"]

    by_path = {r["path"]: r for r in records}
    distinct = [
        by_path["Training/meningioma/img_000.jpg"]["pseudo_patient_id"],
        by_path["Training/meningioma/img_001.jpg"]["pseudo_patient_id"],
        by_path["Testing/notumor/img_000.jpg"]["pseudo_patient_id"],
    ]
    assert len(set(distinct)) == 3, "structurally different images must not be merged"


def test_audit_is_reproducible(planted_dataset: Path) -> None:
    first = audit(planted_dataset, phash_threshold=5, show_progress=False)["report"]
    second = audit(planted_dataset, phash_threshold=5, show_progress=False)["report"]

    assert first["totals"] == second["totals"]
    assert first["cross_split_leakage"]["leaked_component_ids"] == (
        second["cross_split_leakage"]["leaked_component_ids"]
    )
