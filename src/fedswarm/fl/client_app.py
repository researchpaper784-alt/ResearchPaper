"""Phase 3.1 -- the Flower ClientApp. Strategy-agnostic by construction (plan Step
3.1): every strategy-specific behavior (currently just FedProx's `mu`) is read from the
train `ConfigRecord` the server attaches to each round's message, so adding a new
strategy in Phase 5 means a new `Strategy` on the server, not a new client.

Verified against the installed `flwr==1.36.0` API (`docs/FLOWER_API_NOTES.md`), not
against older `fl.client.NumPyClient` patterns:

    app = ClientApp()
    @app.train()
    def train(msg: Message, context: Context) -> Message: ...

Per-round overrides (server_round, local_epochs, lr, mu) arrive in
`msg.content["config"]`; static, run-wide settings (which model, which partition
regime, image size, ...) arrive in `context.run_config`. This split is confirmed by
reading the real `@flwrlabs/quickstart-pytorch` reference app (FLOWER_API_NOTES.md), not
assumed.

⚠️ Not run end-to-end on this machine: `flwr[simulation]` has no wheel for Intel macOS
(`docs/FLOWER_API_NOTES.md`), so there is no local `Grid`/simulation backend to actually
route a `Message` through the Flower runtime here. What *is* verified locally: every
handler below is a plain, undecorated-in-effect function (confirmed by reading
`ClientApp.train`'s decorator source -- it registers and returns the function
unmodified), so `tests/test_client_app.py` calls `train(...)`/`evaluate(...)` directly
with hand-built `Message`/`Context` objects, exercising the exact code path Flower's
runtime would call, minus the runtime's own message routing. The first real
`flwr run` on Colab is what confirms routing, node/run config wiring end-to-end, and
timing -- not something this file can self-certify.
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedswarm.fl.task import build_model_from_run_config, load_client_data, local_evaluate, local_train

app = ClientApp()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _partition_id(context: Context) -> int:
    return int(context.node_config["partition-id"])


@app.train()
def train(msg: Message, context: Context) -> Message:
    model = build_model_from_run_config(context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    train_loader, _ = load_client_data(_partition_id(context), context.run_config)

    config = msg.content["config"]
    metrics = local_train(
        model,
        train_loader,
        _device(),
        epochs=int(config.get("local-epochs", context.run_config.get("local-epochs", 2))),
        lr=float(config.get("lr", context.run_config.get("local-lr", 0.01))),
        mu=float(config.get("mu", 0.0)),
    )
    metrics["client_id"] = _partition_id(context)
    metrics["server_round"] = int(config.get("server_round", -1))

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model = build_model_from_run_config(context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    _, val_loader = load_client_data(_partition_id(context), context.run_config)
    result = local_evaluate(model, val_loader, _device())

    # MetricRecord values are scalars (bool | int | float | str, or lists of those) --
    # verified via ConfigRecord/MetricRecord's constructor signature, which is why the
    # per-class confusion counts are flattened into individually-keyed scalars here
    # rather than nested as `{"per_class_counts": {...}}`.
    flat_metrics: dict = {
        "num-examples": result["num-examples"],
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "client_id": _partition_id(context),
    }
    for cls, counts in result["per_class_counts"].items():
        for key, value in counts.items():
            flat_metrics[f"{cls}_{key}"] = value

    content = RecordDict({"metrics": MetricRecord(flat_metrics)})
    return Message(content, reply_to=msg)


__all__ = ["app", "train", "evaluate"]
