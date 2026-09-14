"""Tests for source-provenance matching (recovering source_component for source_shift
partitioning). Uses synthetic fixtures throughout -- the real Figshare/SARTAJ/Br35H
downloads are gigabytes and not something tests should depend on."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fedswarm.data.source_provenance import (
    HASH_SIZE,
    _decode_pid,
    _phash_bitstring,
    _phash_to_bits,
    _to_uint8_image,
    load_figshare,
    load_flat_image_dir,
    match_manifest_to_sources,
)


def _write_synthetic_mat(path: Path, patient_id: str, label: int, image: np.ndarray) -> None:
    """A minimal cjdata-struct .mat file in the same v7.3/HDF5 layout as the real
    Figshare files, with the same column-major storage MATLAB uses (h5py preserves
    whatever axis order is written, so writing image.T here reproduces the real file's
    on-disk layout -- the load path un-transposes it back)."""
    pid_codes = np.array([[ord(c)] for c in patient_id], dtype=np.uint16)
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g.create_dataset("PID", data=pid_codes)
        g.create_dataset("label", data=np.array([[float(label)]]))
        g.create_dataset("image", data=image.T)  # written transposed, as MATLAB does
        g.create_dataset("tumorMask", data=np.zeros_like(image))


def test_decode_pid_reads_char_codes() -> None:
    raw = np.array([[ord(c)] for c in "100360"], dtype=np.uint16)
    assert _decode_pid(raw) == "100360"


def test_to_uint8_image_min_max_normalizes() -> None:
    raw = np.array([[0, 100], [200, 3366]], dtype=np.int16)
    img = _to_uint8_image(raw)
    arr = np.asarray(img)
    assert arr.min() == 0
    assert arr.max() == 255
    assert img.mode == "L"


def test_to_uint8_image_handles_constant_image() -> None:
    """A flat (all-equal) raw image must not divide by zero."""
    raw = np.full((4, 4), 7, dtype=np.int16)
    img = _to_uint8_image(raw)
    assert np.all(np.asarray(img) == 0)


def test_phash_bitstring_is_64_chars_of_0_or_1() -> None:
    img = Image.new("L", (64, 64), 128)
    bitstring = _phash_bitstring(img)
    assert len(bitstring) == HASH_SIZE * HASH_SIZE
    assert set(bitstring) <= {"0", "1"}


def test_phash_bitstring_matches_manifest_format() -> None:
    """The exact contract this module depends on: our bitstring format must be
    byte-for-byte what dedup.py persists into manifest.csv's phash column, or matching
    silently compares apples to oranges."""
    import imagehash

    img = Image.new("L", (64, 64), 200)
    ours = _phash_bitstring(img)
    theirs = "".join(str(int(b)) for b in imagehash.phash(img, hash_size=HASH_SIZE).hash.flatten())
    assert ours == theirs


def test_load_figshare_recovers_patient_and_label(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 3000, size=(64, 64)).astype(np.int16)
    _write_synthetic_mat(tmp_path / "1.mat", patient_id="100360", label=2, image=image)

    records = load_figshare(tmp_path, show_progress=False)

    assert len(records) == 1
    assert records[0]["source"] == "figshare"
    assert records[0]["patient_id"] == "100360"
    assert records[0]["label"] == "glioma"  # label code 2
    assert len(records[0]["phash"]) == HASH_SIZE * HASH_SIZE


def test_load_figshare_transposes_correctly(tmp_path: Path) -> None:
    """The core bug this module hit: without un-transposing, the recovered image is
    read sideways and its hash bears no resemblance to the real image's hash. This test
    plants a genuinely asymmetric image and checks the hash matches what hashing the
    correctly-oriented image directly would give -- not its rotated self."""
    image = np.zeros((32, 32), dtype=np.int16)
    image[:8, :] = 3000  # a bright band across the top only -- breaks row/col symmetry
    _write_synthetic_mat(tmp_path / "1.mat", patient_id="1", label=1, image=image)

    records = load_figshare(tmp_path, show_progress=False)
    expected = _phash_bitstring(_to_uint8_image(image))

    assert records[0]["phash"] == expected


def test_load_figshare_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_figshare(tmp_path, show_progress=False)


def test_load_flat_image_dir_uses_parent_folder_as_label(tmp_path: Path) -> None:
    (tmp_path / "glioma").mkdir()
    Image.new("L", (32, 32), 100).save(tmp_path / "glioma" / "img1.jpg")

    records = load_flat_image_dir(tmp_path, source_name="sartaj", show_progress=False)

    assert len(records) == 1
    assert records[0]["source"] == "sartaj"
    assert records[0]["label"] == "glioma"
    assert records[0]["patient_id"] is None


def test_phash_to_bits_round_trips() -> None:
    bitstrings = ["1010" * 16, "0000" * 16, "1111" * 16]
    bits = _phash_to_bits(bitstrings)

    assert bits.shape == (3, 64)
    assert bits[1].sum() == 0
    assert bits[2].sum() == 64


# --- matching ---------------------------------------------------------------------------


def _bitstring(pattern: str, length: int = 64) -> str:
    return (pattern * length)[:length]


def test_match_finds_exact_match() -> None:
    manifest = pd.DataFrame({"phash": [_bitstring("1010")]})
    upstream = [{"source": "figshare", "source_file": "a.mat", "phash": _bitstring("1010")}]

    result = match_manifest_to_sources(manifest, upstream, max_distance=5)

    assert result["matched"].iloc[0]
    assert result["source_component"].iloc[0] == "figshare"
    assert result["source_match_distance"].iloc[0] == 0


def test_match_respects_threshold() -> None:
    close = _bitstring("0" * 64)
    far = "1" * 64  # maximally different: distance 64
    manifest = pd.DataFrame({"phash": [close]})
    upstream = [{"source": "figshare", "source_file": "a.mat", "phash": far}]

    result = match_manifest_to_sources(manifest, upstream, max_distance=5)

    assert not result["matched"].iloc[0]
    assert result["source_component"].iloc[0] == "unmatched"
    assert result["source_match_distance"].iloc[0] == 64


def test_match_picks_the_nearest_of_several_candidates() -> None:
    target = "0" * 64
    manifest = pd.DataFrame({"phash": [target]})
    upstream = [
        {"source": "sartaj", "source_file": "far.jpg", "phash": "1" * 10 + "0" * 54},
        {"source": "figshare", "source_file": "near.mat", "phash": "1" * 2 + "0" * 62},
        {"source": "br35h", "source_file": "mid.jpg", "phash": "1" * 5 + "0" * 59},
    ]

    result = match_manifest_to_sources(manifest, upstream, max_distance=10)

    assert result["source_component"].iloc[0] == "figshare"
    assert result["source_match_distance"].iloc[0] == 2


def test_match_requires_phash_column() -> None:
    manifest = pd.DataFrame({"path": ["x.jpg"]})
    with pytest.raises(ValueError, match="phash"):
        match_manifest_to_sources(manifest, [], max_distance=5)


def test_match_handles_empty_manifest_gracefully() -> None:
    manifest = pd.DataFrame({"phash": []})
    upstream = [{"source": "figshare", "source_file": "a.mat", "phash": _bitstring("1")}]

    result = match_manifest_to_sources(manifest, upstream, max_distance=5)

    assert len(result) == 0
