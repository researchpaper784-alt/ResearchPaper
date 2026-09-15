"""Mirror expensive pipeline artifacts to durable storage, stage by stage.

Colab is the training environment (Kaggle needs phone verification we don't have), and
its free tier recycles the runtime when the GPU quota runs out -- wiping all of
`/content` with no warning. A session was lost this way on 2026-09-15: uploaded dataset,
built cache, and finished runs all gone, because nothing left the ephemeral disk until
the very end.

The fix is to treat each expensive stage as independently restorable. Cost ordering, and
why each stage earns a slot:

  dataset  164MB upload over a home connection, the slowest step by far
  cache    a few minutes of JPEG decoding, and needs `dataset` present to rebuild
  results  GPU-hours; a finished run must never be recomputed

Restoring `cache` alone is enough to train: `run_experiment.py` reads the committed
manifest and the cache array, never the raw images. `dataset` is still worth keeping for
the ResNet table, which needs a second cache built at 224.

State is derived from the filesystem rather than tracked in a status file, so it cannot
drift from what is actually on disk -- a stage is done when its artifact exists.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Stage name -> path relative to the repo root. Ordered by position in the pipeline.
ARTIFACTS: dict[str, Path] = {
    "dataset": Path("data/raw/dataset.zip"),
    "cache": Path("data/processed/cache"),
    "results": Path("results/centralized"),
}


class Checkpoint:
    """Copies pipeline artifacts between the working tree and a durable backup directory.

    `backup_dir` is somewhere that survives the runtime (a mounted Drive folder on Colab);
    `root` is the repo working tree.
    """

    def __init__(self, backup_dir: Path | str, root: Path | str = ".") -> None:
        self.backup_dir = Path(backup_dir)
        self.root = Path(root)

    def _pair(self, stage: str) -> tuple[Path, Path]:
        relative = ARTIFACTS[stage]
        return self.root / relative, self.backup_dir / relative

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def save(self, *stages: str) -> list[str]:
        """Back up the named stages (default: all). Stages with no local artifact are
        skipped rather than raising -- callers checkpoint opportunistically after each
        step, and early steps legitimately have nothing to save yet."""
        saved = []
        for stage in stages or ARTIFACTS:
            local, backup = self._pair(stage)
            if local.exists():
                self._copy(local, backup)
                saved.append(stage)
        return saved

    def restore(self, *stages: str) -> list[str]:
        """Bring the named stages (default: all) back into the working tree."""
        restored = []
        for stage in stages or ARTIFACTS:
            local, backup = self._pair(stage)
            if backup.exists():
                self._copy(backup, local)
                restored.append(stage)
        return restored

    def status(self) -> dict[str, dict[str, bool]]:
        return {
            stage: {"local": local.exists(), "backup": backup.exists()}
            for stage in ARTIFACTS
            for local, backup in [self._pair(stage)]
        }

    def report(self) -> str:
        lines = ["stage      local  backup"]
        for stage, state in self.status().items():
            local = "yes" if state["local"] else "--"
            backup = "yes" if state["backup"] else "--"
            lines.append(f"{stage:<10} {local:<6} {backup}")

        results = self.root / ARTIFACTS["results"]
        if results.is_dir():
            done = sorted(p.stem for p in results.glob("*.json"))
            lines.append(f"\n{len(done)} finished runs: {', '.join(done) if done else '(none)'}")
        return "\n".join(lines)
