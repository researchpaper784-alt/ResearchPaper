"""Phase 1.1 — acquire and verify the Brain Tumor MRI dataset.

Two acquisition paths are supported, because the Kaggle API needs credentials that not
every environment has:

  * Kaggle API:  python -m fedswarm.data.download --download
  * Manual zip:  python -m fedswarm.data.download --zip ~/Downloads/archive.zip

On Kaggle itself the dataset is already mounted and neither is needed -- just point
FEDSWARM_DATA_ROOT at the mount and run --verify.

Every count written to DATASET_CARD.md is computed from disk. Nothing is copied from the
dataset's Kaggle page, a blog post, or the implementation plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SPLITS = ("Training", "Testing")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

DEFAULT_ROOT = Path("data/raw/brain-tumor-mri")
CARD_PATH = Path("data/raw/DATASET_CARD.md")


def resolve_root(explicit: str | Path | None = None) -> Path:
    """Dataset root, in priority order: explicit argument, FEDSWARM_DATA_ROOT, default.

    The env var is what makes the same committed manifest resolve against both a local
    download and Kaggle's /kaggle/input/... mount.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("FEDSWARM_DATA_ROOT")
    return Path(env) if env else DEFAULT_ROOT


def find_split_parent(root: Path) -> Path:
    """Locate the directory that actually contains Training/ and Testing/.

    Archives and Kaggle mounts sometimes nest the payload one or two levels deep, so
    search rather than assuming.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}\n"
            f"Fetch it with --download (needs Kaggle credentials) or --zip <path>, "
            f"or set FEDSWARM_DATA_ROOT to an existing copy."
        )

    candidates = [root, *(p for p in sorted(root.rglob("*")) if p.is_dir())]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SPLITS):
            return candidate

    raise FileNotFoundError(
        f"Could not find a directory containing {list(SPLITS)} under {root}. "
        f"Check that the archive extracted correctly."
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_via_kaggle_api(dest_dir: Path) -> Path:
    """Download the dataset archive with the Kaggle CLI. Returns the archive path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_json or has_env):
        raise RuntimeError(
            "No Kaggle credentials found.\n"
            "  Option A: download kaggle.json from https://www.kaggle.com/settings "
            "(Account -> API -> Create New Token) and save it to ~/.kaggle/kaggle.json "
            "with chmod 600.\n"
            "  Option B: set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            f"  Option C: download the zip manually from {DATASET_URL} and run "
            "`--zip <path-to-zip>`."
        )

    cmd = ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest_dir)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    archives = sorted(dest_dir.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"Kaggle CLI reported success but no .zip appeared in {dest_dir}")
    return max(archives, key=lambda p: p.stat().st_mtime)


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def scan_counts(split_parent: Path) -> dict[str, dict[str, int]]:
    """Per-split, per-class image counts, computed by listing files on disk."""
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            class_dir = split_parent / split / cls
            if not class_dir.is_dir():
                counts[split][cls] = 0
                continue
            counts[split][cls] = sum(
                1 for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
    return counts


def scan_image_properties(split_parent: Path) -> dict[str, Any]:
    """Observed image size and colour-mode distribution.

    Reads only image headers (PIL is lazy about pixel data), so this is fast enough to
    run over every file rather than a sample.
    """
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    unreadable: list[str] = []

    for split in SPLITS:
        for cls in CLASSES:
            class_dir = split_parent / split / cls
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    with Image.open(path) as img:
                        sizes[f"{img.size[0]}x{img.size[1]}"] += 1
                        modes[img.mode] += 1
                except Exception as exc:  # noqa: BLE001 - want the filename, not a crash
                    unreadable.append(f"{path}: {exc}")

    return {
        "sizes": dict(sizes.most_common()),
        "modes": dict(modes.most_common()),
        "unreadable": unreadable,
    }


def render_counts_table(counts: dict[str, dict[str, int]]) -> str:
    header = "| Split | " + " | ".join(CLASSES) + " | **Total** |"
    sep = "|---" * (len(CLASSES) + 2) + "|"
    lines = [header, sep]
    for split in SPLITS:
        row = counts[split]
        total = sum(row.values())
        lines.append(
            f"| {split} | " + " | ".join(str(row[c]) for c in CLASSES) + f" | **{total}** |"
        )
    grand = {c: sum(counts[s][c] for s in SPLITS) for c in CLASSES}
    lines.append(
        "| **Total** | "
        + " | ".join(f"**{grand[c]}**" for c in CLASSES)
        + f" | **{sum(grand.values())}** |"
    )
    return "\n".join(lines)


def write_dataset_card(
    card_path: Path,
    split_parent: Path,
    counts: dict[str, dict[str, int]],
    properties: dict[str, Any],
    archive: Path | None,
) -> None:
    archive_sha = sha256_file(archive) if archive and archive.exists() else None
    archive_size = archive.stat().st_size if archive and archive.exists() else None

    size_lines = "\n".join(
        f"| `{size}` | {n} |" for size, n in list(properties["sizes"].items())[:20]
    )
    n_distinct_sizes = len(properties["sizes"])
    mode_lines = "\n".join(f"| `{mode}` | {n} |" for mode, n in properties["modes"].items())

    unreadable = properties["unreadable"]
    unreadable_section = (
        "\n".join(f"- `{u}`" for u in unreadable) if unreadable else "None — every file opened cleanly."
    )

    card = f"""# Dataset card — Brain Tumor MRI

**Every number in this file was computed from disk** by
`python -m fedswarm.data.download --verify`. Nothing here is copied from the dataset's
Kaggle page, a blog post, or the implementation plan. Re-run that command to regenerate.

| Field | Value |
|---|---|
| Source URL | {DATASET_URL} |
| Kaggle slug | `{DATASET_SLUG}` |
| Verified (UTC) | {datetime.now(timezone.utc).isoformat()} |
| Scanned path | `{split_parent}` |
| Archive SHA256 | {f"`{archive_sha}`" if archive_sha else "n/a — verified against an already-extracted copy, no archive present"} |
| Archive size (bytes) | {archive_size if archive_size is not None else "n/a"} |
| License | **UNVERIFIED — read it from {DATASET_URL} and record it here before the paper cites this dataset.** |

## Provenance

This dataset is a merge of three sources, per its Kaggle description: the figshare brain
tumor dataset (Cheng et al.), the SARTAJ dataset, and Br35H. That multi-source structure
is what Phase 1.4's `source_shift` partitioning regime exploits to simulate realistic
cross-site feature shift.

**Known issue to carry into the paper's limitations:** the SARTAJ component has documented
mislabeling in the glioma class. Phase 1.2 records this but does not attempt to fix it.

## Per-class, per-split counts (counted from disk)

{render_counts_table(counts)}

⚠️ These are the counts of the **original, as-distributed split**, before de-duplication.
Phase 1.2 rebuilds the split at the pseudo-patient level; the counts that belong in the
paper are the post-de-duplication ones in `data/processed/leakage_report.json`.

## Observed image sizes

{n_distinct_sizes} distinct sizes observed. {"Top 20 shown." if n_distinct_sizes > 20 else ""}

| Size (WxH) | Count |
|---|---|
{size_lines}

## Observed colour modes

| PIL mode | Count |
|---|---|
{mode_lines}

## Unreadable files

{unreadable_section}
"""
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card)


def verify(root: Path, archive: Path | None = None, card_path: Path = CARD_PATH) -> dict[str, Any]:
    split_parent = find_split_parent(root)
    print(f"Scanning {split_parent} ...")

    counts = scan_counts(split_parent)
    properties = scan_image_properties(split_parent)

    print()
    print(render_counts_table(counts))
    print()
    print(f"Distinct image sizes: {len(properties['sizes'])}")
    print(f"Colour modes: {properties['modes']}")
    if properties["unreadable"]:
        print(f"⚠️  {len(properties['unreadable'])} unreadable file(s)")

    write_dataset_card(card_path, split_parent, counts, properties, archive)
    print(f"\nWrote {card_path}")

    return {"split_parent": str(split_parent), "counts": counts, "properties": properties}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Dataset root (default: $FEDSWARM_DATA_ROOT)")
    parser.add_argument("--download", action="store_true", help="Fetch via the Kaggle API")
    parser.add_argument("--zip", default=None, help="Extract a manually downloaded archive")
    parser.add_argument("--verify", action="store_true", help="Scan from disk and write the card")
    parser.add_argument("--counts-json", default=None, help="Also write counts to this JSON path")
    args = parser.parse_args()

    root = resolve_root(args.root)
    archive: Path | None = None

    try:
        if args.download:
            archive = download_via_kaggle_api(Path("data/raw"))
            extract_archive(archive, root)

        if args.zip:
            archive = Path(args.zip)
            if not archive.exists():
                raise FileNotFoundError(f"No such archive: {archive}")
            extract_archive(archive, root)

        if args.verify or args.download or args.zip:
            result = verify(root, archive=archive)
            if args.counts_json:
                Path(args.counts_json).write_text(json.dumps(result["counts"], indent=2))
        else:
            parser.print_help()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
