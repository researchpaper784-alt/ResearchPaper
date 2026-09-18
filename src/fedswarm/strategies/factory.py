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

from fedswarm.aco.colony import ColonyConfig
from fedswarm.aco.fitness import DataFreeFitnessConfig
from fedswarm.aco.heuristics import HeuristicWeights
from fedswarm.aco.pheromone import PheromoneConfig
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
        return _build_fedaco(run_config, common, model=model, val_loader=val_loader, device=device)

    raise ValueError(f"Unknown strategy-name: {name!r}")


def _build_fedaco(
    run_config: RunConfig,
    common: dict,
    *,
    model: torch.nn.Module | None,
    val_loader: DataLoader | None,
    device: torch.device | None,
) -> FedACO:
    """Every field below is a Phase 7 ablation axis (plan §7): fedaco-fitness-mode is
    A3, fedaco-pheromone-persistence/-rho-round are A2, fedaco-gamma-* are A4,
    fedaco-beta-* are A5, fedaco-{a,b,q0,rho,ants,iters,levels} are A6,
    fedaco-target-sum is A7. Defaults match the plan's §14 hyperparameter table where
    that table specifies one; the rest (trim fraction, level bounds, tau bounds) are
    this repo's own already-tested defaults."""
    aco_config = FedACOConfig(
        num_levels=int(run_config.get("fedaco-num-levels", 11)),
        level_low=float(run_config.get("fedaco-level-low", 0.0)),
        level_high=float(run_config.get("fedaco-level-high", 2.5)),
        target_sum=float(run_config.get("fedaco-target-sum", 1.0)),
        num_rounds=int(run_config.get("num-rounds", 2)),
        ants_start=int(run_config.get("fedaco-ants-start", 30)),
        ants_end=int(run_config.get("fedaco-ants-end", 10)),
        iters_start=int(run_config.get("fedaco-iters-start", 10)),
        iters_end=int(run_config.get("fedaco-iters-end", 4)),
        trim_fraction=float(run_config.get("fedaco-trim-fraction", 0.2)),
        safety_fallback=bool(run_config.get("fedaco-safety-fallback", True)),
        fitness_mode=str(run_config.get("fedaco-fitness-mode", "data_free")),  # type: ignore[arg-type]
        search_method=str(run_config.get("fedaco-search-method", "aco")),  # type: ignore[arg-type]
        colony=ColonyConfig(
            pheromone_exp=float(run_config.get("fedaco-a-exponent", 1.0)),
            heuristic_exp=float(run_config.get("fedaco-b-exponent", 2.0)),
            q0=float(run_config.get("fedaco-q0", 0.9)),
            rho=float(run_config.get("fedaco-rho", 0.1)),
            tau_min=float(run_config.get("fedaco-tau-min", 0.01)),
            tau_max=float(run_config.get("fedaco-tau-max", 10.0)),
        ),
        pheromone=PheromoneConfig(
            tau_min=float(run_config.get("fedaco-tau-min", 0.01)),
            tau_max=float(run_config.get("fedaco-tau-max", 10.0)),
            tau0=float(run_config.get("fedaco-tau0", 1.0)),
            rho_round=float(run_config.get("fedaco-rho-round", 0.1)),
            persistence=str(run_config.get("fedaco-pheromone-persistence", "decayed")),  # type: ignore[arg-type]
        ),
        fitness=DataFreeFitnessConfig(
            gamma_alignment=float(run_config.get("fedaco-gamma-alignment", 1.0)),
            gamma_dispersion=float(run_config.get("fedaco-gamma-dispersion", 0.5)),
            gamma_entropy=float(run_config.get("fedaco-gamma-entropy", 0.1)),
        ),
        heuristics=HeuristicWeights(
            beta_alignment=float(run_config.get("fedaco-beta-alignment", 2.0)),
            beta_drift=float(run_config.get("fedaco-beta-drift", 1.0)),
            beta_val_improvement=float(run_config.get("fedaco-beta-val-improvement", 1.0)),
            beta_data_size=float(run_config.get("fedaco-beta-data-size", 0.5)),
        ),
    )
    return FedACO(**common, aco_config=aco_config, model=model, val_loader=val_loader, device=device)
