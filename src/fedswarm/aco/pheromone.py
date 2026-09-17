"""Cross-round pheromone persistence (plan §4.3) -- the "Adaptive" in FedACO/FedSwarm.

Stigmergy: persistent shared memory deposited in the environment. Keeping tau across
rounds, keyed by client identity rather than by position in that round's sampled set,
means the colony starts each round from an informed prior instead of from scratch --
an accumulated record of which clients have historically produced useful updates. This
is the mechanism the paper's Related Work section and the Phase 7 persistence ablation
both hinge on: remove it (`persistence="none"`) and the "swarm intelligence" framing
should show a measurable drop, or the framing is decorative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Persistence = Literal["none", "full", "decayed"]


@dataclass
class PheromoneConfig:
    tau_min: float = 0.01
    tau_max: float = 10.0
    tau0: float = 1.0
    rho_round: float = 0.1  # cross-round decay toward tau0, used when persistence="decayed"
    absence_decay: float = 0.5  # extra multiplicative decay per round a client is absent
    persistence: Persistence = "decayed"


class Pheromone:
    """tau, stored as one row (length `num_levels`) per client id, not as a single
    K x L matrix -- the participating client set S_t changes round to round, but a
    client's row must survive across rounds it doesn't participate in at all."""

    def __init__(self, num_levels: int, config: PheromoneConfig | None = None) -> None:
        self.num_levels = num_levels
        self.config = config or PheromoneConfig()
        self._rows: dict[str, torch.Tensor] = {}
        self._rounds_absent: dict[str, int] = {}

    def _new_row(self) -> torch.Tensor:
        return torch.full((self.num_levels,), self.config.tau0)

    def begin_round(self, client_ids: list[str]) -> torch.Tensor:
        """Applies cross-round persistence for every client about to participate this
        round, initializes rows for newly-seen clients at tau0, and returns the
        [K, num_levels] matrix a colony run operates on (K = len(client_ids), in the
        given order)."""
        present = set(client_ids)
        for client_id in list(self._rows.keys()):
            if client_id in present or self.config.persistence == "full":
                continue
            # Absent clients decay toward tau0 regardless of "none" vs "decayed" --
            # those two only differ in how a *participating* client's row is treated
            # after its own round (see `end_round`); a client that simply doesn't show
            # up this round shouldn't jump discontinuously between the two modes.
            self._rounds_absent[client_id] = self._rounds_absent.get(client_id, 0) + 1
            decay = self.config.absence_decay ** self._rounds_absent[client_id]
            self._rows[client_id] = self._rows[client_id] * decay + self._new_row() * (1.0 - decay)

        rows = []
        for client_id in client_ids:
            if client_id not in self._rows:
                self._rows[client_id] = self._new_row()
            self._rounds_absent[client_id] = 0
            rows.append(self._rows[client_id])
        return torch.stack(rows, dim=0).clone()

    def end_round(self, client_ids: list[str], tau_final: torch.Tensor) -> None:
        """Writes a colony run's post-evaporation/deposit tau back per client id,
        applying the configured cross-round decay toward tau0 on top."""
        for idx, client_id in enumerate(client_ids):
            row = tau_final[idx]
            if self.config.persistence == "none":
                row = self._new_row()
            elif self.config.persistence == "decayed":
                row = (1.0 - self.config.rho_round) * row + self.config.rho_round * self.config.tau0
            row = row.clamp(self.config.tau_min, self.config.tau_max)
            self._rows[client_id] = row

    def entropy(self, client_ids: list[str]) -> float:
        """Mean per-client normalized pheromone entropy -- a convergence diagnostic
        (plan §4.7): high entropy means the colony still spreads desirability broadly
        across levels; low entropy means it has concentrated on a few."""
        matrix = torch.stack([self._rows[c] for c in client_ids if c in self._rows], dim=0)
        probs = (matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
        row_entropy = -(probs * probs.log()).sum(dim=1)
        return float(row_entropy.mean())
