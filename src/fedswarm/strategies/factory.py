"""Strategy factory (plan §5): selects among FedAvg and its baselines by
`run_config["strategy-name"]`, so `fl/app.py`'s `server_app.main()` (and any future
sweep runner) drives every strategy through one entry point instead of a bespoke
branch per strategy.

Built-ins (`flwr.serverapp.strategy`) preferred over a custom reimplementation
wherever the installed `flwr==1.36.0` provides one -- verified real constructors via
the installed package's own source, not assumed; see `docs/FLOWER_API_NOTES.md`.
Everything else lives in `fedswarm/strategies/`.
"""

from __future__ import annotations

import torch
from flwr.serverapp.strategy import FedAdam, FedAvg, FedMedian, FedProx, FedTrimmedAvg, FedYogi, Krum, Strategy
from torch.utils.data import DataLoader

from fedswarm.strategies.fedaco import FedACO, FedACOConfig
from fedswarm.strategies.fedlaw import FedLAW
from fedswarm.strategies.fednova import FedNova
from fedswarm.strategies.lossbased import LossBasedWeighting
from fedswarm.strategies.scaffold import Scaffold

RunConfig = dict


def strategy_from_run_config(
    run_config: RunConfig,
    *,
    model: torch.nn.Module | None = None,
    val_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> Strategy:
    """`model`/`val_loader`/`device` are only required for `strategy-name="fedlaw"`
    (needs a differentiable forward pass against a server val batch each round) --
    every other strategy ignores them, so callers not using FedLAW can omit them."""
    common = dict(
        fraction_train=float(run_config.get("fraction-train", 1.0)),
        fraction_evaluate=float(run_config.get("fraction-evaluate", 1.0)),
        min_train_nodes=int(run_config.get("min-train-nodes", 2)),
        min_evaluate_nodes=int(run_config.get("min-evaluate-nodes", 2)),
        min_available_nodes=int(run_config.get("min-available-nodes", 2)),
    )
    name = str(run_config.get("strategy-name", "fedavg")).lower()

    if name == "fedavg":
        return FedAvg(**common)

    if name == "fedprox":
        return FedProx(**common, proximal_mu=float(run_config.get("fedprox-mu", 0.01)))

    if name in ("fedadam", "fedyogi"):
        opt_defaults = {"fedadam": (0.1, 0.1), "fedyogi": (0.01, 0.0316)}[name]
        eta_default, eta_l_default = opt_defaults
        cls = FedAdam if name == "fedadam" else FedYogi
        return cls(
            **common,
            eta=float(run_config.get("fedopt-eta", eta_default)),
            eta_l=float(run_config.get("fedopt-eta-l", eta_l_default)),
            beta_1=float(run_config.get("fedopt-beta1", 0.9)),
            beta_2=float(run_config.get("fedopt-beta2", 0.99)),
            tau=float(run_config.get("fedopt-tau", 0.001)),
        )

    if name in ("krum", "multikrum"):
        return Krum(**common, num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)))

    if name in ("trimmed-mean", "fedtrimmedavg"):
        return FedTrimmedAvg(**common, beta=float(run_config.get("trim-beta", 0.2)))

    if name == "median":
        return FedMedian(**common)

    if name == "fednova":
        return FedNova(**common)

    if name in ("loss-based", "fednolowe"):
        return LossBasedWeighting(
            **common, temperature=float(run_config.get("lossweight-temperature", 1.0))
        )

    if name == "scaffold":
        return Scaffold(**common, server_lr=float(run_config.get("scaffold-server-lr", 1.0)))

    if name == "fedlaw":
        if model is None or val_loader is None or device is None:
            raise ValueError("strategy-name='fedlaw' requires model, val_loader, and device")
        return FedLAW(
            **common,
            model=model,
            val_loader=val_loader,
            device=device,
            num_steps=int(run_config.get("fedlaw-steps", 20)),
            lr=float(run_config.get("fedlaw-lr", 0.1)),
        )

    if name == "fedaco":
        return FedACO(
            **common,
            aco_config=FedACOConfig(num_rounds=int(run_config.get("num-rounds", 2))),
        )

    raise ValueError(f"Unknown strategy-name: {name!r}")
