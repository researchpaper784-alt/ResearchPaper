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
    (needs a differentiable forward pass against a server val batch each round) and for
    `strategy-name="fedaco"` with `aco-fitness-mode="server_val"` (scores candidate
    alphas against the same server-held set). Every other strategy -- including FedACO
    in its default `data_free` mode, which is the whole point of the method -- ignores
    them, so callers not using those two can omit them."""
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
        # Deliberately per-strategy keys (`fedadam-eta`, `fedyogi-eta`) rather than one
        # shared `fedopt-eta`. FedAdam and FedYogi carry *different* published defaults
        # (0.1/0.1 vs 0.01/0.0316, from the FedOpt paper), and a single declared key
        # would resolve to one literal value for both -- silently replacing FedYogi's
        # default with FedAdam's and making the fallbacks below dead code. Every key
        # here must be declared in pyproject.toml to be overridable at all, and a
        # declared key always has a value, so "fall back to the strategy's own default"
        # only works if the key is unique to that strategy.
        eta_default, eta_l_default = {"fedadam": (0.1, 0.1), "fedyogi": (0.01, 0.0316)}[name]
        cls = FedAdam if name == "fedadam" else FedYogi
        return cls(
            **common,
            eta=float(run_config.get(f"{name}-eta", eta_default)),
            eta_l=float(run_config.get(f"{name}-eta-l", eta_l_default)),
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


def _as_bool(value: object, default: bool) -> bool:
    """`Context.run_config` values are bool|float|int|str. A TOML `false` arrives as a
    real bool, but a value that has been round-tripped through a string ("false", "0")
    would be silently truthy under a bare `bool(...)` -- which, for a flag like
    `aco-safety-fallback`, means an ablation quietly not running. Parse explicitly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean run-config value")
    return default


def _build_fedaco(
    run_config: RunConfig,
    common: dict,
    *,
    model: torch.nn.Module | None,
    val_loader: DataLoader | None,
    device: torch.device | None,
) -> FedACO:
    """Every FedACO knob reachable from `run_config`, on the same `--run-config` footing
    as the baselines' knobs.

    Until 2026-09-17 this passed only `num-rounds` and left everything else at its
    dataclass default, which made two Phase 7 ablations literally unrunnable: the
    persistence ablation (`aco-persistence="none"` is implemented and unit-tested in
    `aco/pheromone.py` but had no way in) and the global-shrinkage sweep that
    `docs/OPEN_QUESTIONS.md` defers to Phase 7 by promising `target_sum` would be swept
    rather than searched. Phase 6's sweep had nothing to vary either. Keys are
    `aco-`-prefixed to keep them distinguishable from the baselines' in a flat
    `Context.run_config` (which is dict[str, bool|float|int|str] only -- no nesting).
    """
    colony = ColonyConfig(
        pheromone_exp=float(run_config.get("aco-pheromone-exp", ColonyConfig.pheromone_exp)),
        heuristic_exp=float(run_config.get("aco-heuristic-exp", ColonyConfig.heuristic_exp)),
        q0=float(run_config.get("aco-q0", ColonyConfig.q0)),
        rho=float(run_config.get("aco-rho", ColonyConfig.rho)),
        q_deposit=float(run_config.get("aco-q-deposit", ColonyConfig.q_deposit)),
        tau_min=float(run_config.get("aco-tau-min", ColonyConfig.tau_min)),
        tau_max=float(run_config.get("aco-tau-max", ColonyConfig.tau_max)),
        global_best_every=int(
            run_config.get("aco-global-best-every", ColonyConfig.global_best_every)
        ),
        stagnation_patience=int(
            run_config.get("aco-stagnation-patience", ColonyConfig.stagnation_patience)
        ),
    )
    persistence = str(run_config.get("aco-persistence", PheromoneConfig.persistence))
    if persistence not in ("none", "full", "decayed"):
        # Pheromone.end_round branches on this string; an unrecognized value would fall
        # through both branches and silently behave like "full", turning a mistyped
        # ablation into a valid-looking run of the wrong configuration.
        raise ValueError(
            f"Unknown aco-persistence {persistence!r} (expected 'none', 'full', or 'decayed')"
        )
    pheromone = PheromoneConfig(
        tau_min=float(run_config.get("aco-tau-min", PheromoneConfig.tau_min)),
        tau_max=float(run_config.get("aco-tau-max", PheromoneConfig.tau_max)),
        tau0=float(run_config.get("aco-tau0", PheromoneConfig.tau0)),
        rho_round=float(run_config.get("aco-rho-round", PheromoneConfig.rho_round)),
        absence_decay=float(run_config.get("aco-absence-decay", PheromoneConfig.absence_decay)),
        # The Phase 7 A1 ablation: "none" | "full" | "decayed".
        persistence=persistence,
    )
    fitness = DataFreeFitnessConfig(
        gamma_alignment=float(
            run_config.get("aco-gamma-alignment", DataFreeFitnessConfig.gamma_alignment)
        ),
        gamma_dispersion=float(
            run_config.get("aco-gamma-dispersion", DataFreeFitnessConfig.gamma_dispersion)
        ),
        gamma_entropy=float(
            run_config.get("aco-gamma-entropy", DataFreeFitnessConfig.gamma_entropy)
        ),
        normalize_dispersion=_as_bool(
            run_config.get("aco-normalize-dispersion", DataFreeFitnessConfig.normalize_dispersion),
            DataFreeFitnessConfig.normalize_dispersion,
        ),
    )
    heuristics = HeuristicWeights(
        beta_alignment=float(run_config.get("aco-beta-alignment", HeuristicWeights.beta_alignment)),
        beta_drift=float(run_config.get("aco-beta-drift", HeuristicWeights.beta_drift)),
        beta_val_improvement=float(
            run_config.get("aco-beta-val-improvement", HeuristicWeights.beta_val_improvement)
        ),
        beta_data_size=float(run_config.get("aco-beta-data-size", HeuristicWeights.beta_data_size)),
    )
    aco_config = FedACOConfig(
        num_levels=int(run_config.get("aco-num-levels", FedACOConfig.num_levels)),
        level_low=float(run_config.get("aco-level-low", FedACOConfig.level_low)),
        level_high=float(run_config.get("aco-level-high", FedACOConfig.level_high)),
        # Phase 7 sweeps fixed values of the global shrinkage s; it is never searched.
        target_sum=float(run_config.get("aco-target-sum", FedACOConfig.target_sum)),
        num_rounds=int(run_config.get("num-rounds", 2)),
        ants_start=int(run_config.get("aco-ants-start", FedACOConfig.ants_start)),
        ants_end=int(run_config.get("aco-ants-end", FedACOConfig.ants_end)),
        iters_start=int(run_config.get("aco-iters-start", FedACOConfig.iters_start)),
        iters_end=int(run_config.get("aco-iters-end", FedACOConfig.iters_end)),
        trim_fraction=float(run_config.get("aco-trim-fraction", FedACOConfig.trim_fraction)),
        safety_fallback=_as_bool(
            run_config.get("aco-safety-fallback", FedACOConfig.safety_fallback),
            FedACOConfig.safety_fallback,
        ),
        # Shares the run-level seed so the colony is reproducible with everything else.
        seed=int(run_config.get("seed", 0)),
        fitness_mode=str(run_config.get("aco-fitness-mode", FedACOConfig.fitness_mode)),
        colony=colony,
        pheromone=pheromone,
        fitness=fitness,
        heuristics=heuristics,
    )
    return FedACO(
        **common, aco_config=aco_config, model=model, val_loader=val_loader, device=device
    )
