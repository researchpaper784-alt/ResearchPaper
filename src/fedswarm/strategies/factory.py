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
from fedswarm.aco.fitness import (
    CONCENTRATION_PENALTIES,
    DISPERSION_REFERENCES,
    DataFreeFitnessConfig,
)
from fedswarm.aco.heuristics import HeuristicWeights
from fedswarm.fl.attacks import attack_from_run_config, malicious_ids
from fedswarm.aco.pheromone import PheromoneConfig
from fedswarm.strategies.fedaco import FedACO, FedACOConfig
from fedswarm.strategies.fedlaw import FedLAW
from fedswarm.strategies.fednova import FedNova
from fedswarm.strategies.lossbased import LossBasedWeighting
from fedswarm.strategies.scaffold import Scaffold

RunConfig = dict


def _krum_malicious_count(run_config: RunConfig) -> int:
    """How many Byzantine clients Krum should assume -- derived from the attack, not fixed.

    `robustness_r1_label_flip.yaml` and `robustness_r2_update_attack.yaml` both hardcoded
    `num-malicious-nodes: 4` (20% of 20 clients) across attacker fractions of 10%, 20% and
    30%, because a Cartesian-product grid has no way to say "this strategy-specific value
    must track that partition-specific one". Krum was therefore correctly tuned in exactly
    one of the three cells per sweep and mis-tuned in the other two -- and Krum is the
    headline Byzantine baseline the paper's "resilience for free" claim is measured against,
    so a mis-tuned Krum flatters FedACO in a way no reader could detect from the tables.

    The coupling belongs here, where the whole resolved run_config is visible. `f` is taken
    from `attacks.malicious_ids` -- the same function the ClientApp uses to decide which
    partitions actually lie -- so the number Krum assumes and the number of real attackers
    cannot drift apart. `ceil`, at least one attacker for any nonzero fraction, identical
    rule on both sides.

    An explicit `num-malicious-nodes` still wins, so a deliberate "what if Krum
    misjudges f" cell stays expressible. `attack="none"` yields 0 regardless of
    `attack-fraction`, which is what the clean arm of every robustness sweep needs.

    Safe at every cell C runs: Flower's MultiKrum uses
    `num_closest = max(1, n - f - 2)`, so K=20 with f=6 (the 30% cell) leaves 12 neighbours
    and no constraint is violated.
    """
    explicit = run_config.get("num-malicious-nodes")
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    attack, fraction, _ = attack_from_run_config(run_config)
    if attack == "none":
        return 0
    return len(malicious_ids(int(run_config.get("num-clients", 20)), fraction))


def _with_server_round(strategy: Strategy) -> Strategy:
    """Make every strategy put the round number into the train config it sends clients.

    Nothing did. `fl/app.py::train_handler` reads `config.get("server_round", ...)` and
    the ServerApp builds `train_config` once, before round 1, with no round number in it
    -- so the key was never present for any strategy. Scaffold wrote one, spelled
    `server-round`, which no reader uses. Two consequences, found in the first real GPU
    run's metrics:

    1. Every train reply recorded `server_round = -1` (line 549's default), so the
       per-client train metrics in every result file carry no usable round number.
    2. Worse, and the reason this is a correctness fix rather than a cosmetic one: R4's
       Gaussian mechanism seeds its noise with `seed*104729 + partition*1000003 +
       server_round`. With the term pinned at 0, every round drew the **same** noise --
       a fixed per-client perturbation the model trains around, not DP noise. R4 would
       have reported FedACO as far more noise-robust than it is, and the result files
       would carry nothing to reveal it.

    Applied by wrapping rather than in each strategy: `fedavg`, `fedprox`, `fedadam`,
    `fedyogi`, `fedmedian`, `fedtrimmedavg` and `krum` are Flower built-ins with no
    subclass here to override, so a per-strategy fix would have covered only the five
    custom ones -- and silently left the built-ins, which are every baseline the paper
    compares against. `flwr==1.36.0` calls `configure_train(current_round, arrays,
    train_config, grid)` positionally (verified in the installed
    `serverapp/strategy/strategy.py`), and passes the same ConfigRecord object every
    round, so assigning into it per round is both safe and sufficient.
    """
    original = strategy.configure_train

    def configure_train(server_round, arrays, config, grid):  # type: ignore[no-untyped-def]
        config["server_round"] = int(server_round)
        return original(server_round, arrays, config, grid)

    strategy.configure_train = configure_train  # type: ignore[method-assign]
    return strategy


def strategy_from_run_config(
    run_config: RunConfig,
    *,
    model: torch.nn.Module | None = None,
    val_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> Strategy:
    """Wraps `_build_strategy` so every strategy reports its round number to clients.

    `model`/`val_loader`/`device` are only required for `strategy-name="fedlaw"`
    (needs a differentiable forward pass against a server val batch each round) and for
    `strategy-name="fedaco"` with `aco-fitness-mode="server_val"` (scores candidate
    alphas against the same server-held set). Every other strategy -- including FedACO
    in its default `data_free` mode, which is the whole point of the method -- ignores
    them, so callers not using those two can omit them."""
    return _with_server_round(
        _build_strategy(run_config, model=model, val_loader=val_loader, device=device)
    )


def _build_strategy(
    run_config: RunConfig,
    *,
    model: torch.nn.Module | None = None,
    val_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> Strategy:
    """Selection only. `strategy_from_run_config` is the entry point callers use."""
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
        return Krum(**common, num_malicious_nodes=_krum_malicious_count(run_config))

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
    penalty_shape = str(
        run_config.get("aco-concentration-penalty", DataFreeFitnessConfig.concentration_penalty)
    )
    if penalty_shape not in CONCENTRATION_PENALTIES:
        # Validated here rather than at first use: `concentration_penalty` is called once
        # per ant per iteration, so an unknown value would surface from inside the colony
        # mid-round -- where `flwr run` reports it as "Exit Code: 700" and still exits 0.
        raise ValueError(
            f"Unknown aco-concentration-penalty {penalty_shape!r} "
            f"(expected one of {CONCENTRATION_PENALTIES})"
        )
    dispersion_reference = str(
        run_config.get("aco-dispersion-reference", DataFreeFitnessConfig.dispersion_reference)
    )
    if dispersion_reference not in DISPERSION_REFERENCES:
        raise ValueError(
            f"Unknown aco-dispersion-reference {dispersion_reference!r} "
            f"(expected one of {DISPERSION_REFERENCES})"
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
        # Phase 7's penalty-shape ablation: "entropy" (the method as proposed) | "gini".
        concentration_penalty=penalty_shape,
        # "weighted_mean" (the method as proposed) | "base". Validated here for the same
        # reason as the penalty shape: an unknown value would otherwise surface from inside a
        # round, where `flwr run` reports "Exit Code: 700" and still exits 0.
        dispersion_reference=dispersion_reference,
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
        level_spacing=str(
            run_config.get("aco-level-spacing", FedACOConfig.level_spacing)
        ),
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
        # Phase 7 A1. FedACO validates the value at construction, so a typo fails
        # there rather than inside a round.
        search_method=str(
            run_config.get("aco-search-method", FedACOConfig.search_method)
        ),
        colony=colony,
        pheromone=pheromone,
        fitness=fitness,
        heuristics=heuristics,
    )
    return FedACO(
        **common, aco_config=aco_config, model=model, val_loader=val_loader, device=device
    )
