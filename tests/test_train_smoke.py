"""End-to-end smoke test of the training entrypoint on synthetic data.

Exercises the real recipe (AdamW + cosine warmup + label smoothing), the
val/test evaluation path and the metrics.json contract, without the
license-gated dataset.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_scanobjectnn", os.path.join(SCRIPTS, "train_scanobjectnn.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(num_objects, num_classes=15, raw_points=2048, seed=0):
    """Clouds with a class-dependent offset so the model can actually learn."""
    rng = np.random.default_rng(seed)
    labels = np.arange(num_objects) % num_classes
    points = rng.normal(scale=0.3, size=(num_objects, raw_points, 3)).astype(np.float32)
    for i, label in enumerate(labels):
        points[i, :, label % 3] += float(label)
    return points, labels.astype(np.int64)


@pytest.mark.parametrize("recipe", ["current", "pointnext"])
def test_training_runs_and_writes_valid_metrics(tmp_path, monkeypatch, recipe):
    train_mod = _load_train_module()

    train_points, train_labels = _synthetic(120, seed=1)
    test_points, test_labels = _synthetic(45, seed=2)

    def fake_load_split(data_dir, split):
        return (
            (train_points, train_labels) if split == "train"
            else (test_points, test_labels)
        )

    monkeypatch.setattr(train_mod, "load_split", fake_load_split)
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_scanobjectnn.py",
            "--data_dir", "unused",
            "--out", str(tmp_path),
            "--config", "XS",
            "--delta", "0.25",
            "--gmp", "on",
            "--ordering", "morton",
            "--recipe", recipe,
            "--seed", "0",
            "--epochs", "2",
            "--warmup_epochs", "1",
            "--batch_size", "8",
            "--val_fraction", "0.2",
        ],
    )

    train_mod.main()

    with open(tmp_path / "metrics.json") as handle:
        metrics = json.load(handle)

    assert metrics["completed"] is True
    assert metrics["epochs_completed"] == 2
    expected_params = 156_559 if recipe == "current" else 156_623
    assert metrics["params"] == expected_params
    assert "flops" in metrics
    assert metrics["args"]["config"] == "XS"
    assert metrics["args"]["height_append"] is (recipe == "pointnext")
    assert metrics["git_sha"]

    for record in metrics["per_epoch"]:
        for key in ("val_oa", "val_macc", "test_oa", "test_macc"):
            assert 0.0 <= record[key] <= 1.0
        assert np.isfinite(record["train_loss"])

    # Headline numbers and the descriptive curve max are all present and
    # clearly separated.
    assert 0.0 <= metrics["at_best_val"]["test_oa"] <= 1.0
    assert 0.0 <= metrics["final"]["test_oa"] <= 1.0
    assert metrics["test_curve_max"]["test_oa"] >= metrics["final"]["test_oa"] - 1e-9

    assert (tmp_path / "best_val.weights.h5").exists()
    assert (tmp_path / "final.weights.h5").exists()


def test_current_recipe_defaults_are_the_historical_values():
    train_mod = _load_train_module()
    args = train_mod.parse_args(["--data_dir", "unused", "--out", "unused"])
    assert vars(args) == {
        "data_dir": "unused", "out": "unused", "config": "S", "delta": 0.25,
        "gmp": "on", "ordering": "morton", "patch_size": None, "seed": 0,
        "recipe": "current", "epochs": 250, "batch_size": 32, "lr": 1e-3,
        "weight_decay": 0.05, "warmup_epochs": 10, "label_smoothing": 0.2,
        "height_append": False, "val_fraction": 0.1,
    }


def test_pointnext_recipe_selects_full_bundle_and_allows_explicit_overrides():
    train_mod = _load_train_module()
    args = train_mod.parse_args([
        "--data_dir", "unused", "--out", "unused", "--recipe", "pointnext"
    ])
    assert (args.epochs, args.batch_size, args.lr, args.weight_decay) == (
        250, 32, 1e-3, 0.05
    )
    assert (args.warmup_epochs, args.label_smoothing, args.height_append) == (
        0, 0.2, True
    )
    overridden = train_mod.parse_args([
        "--data_dir", "unused", "--out", "unused", "--recipe", "pointnext",
        "--epochs", "2", "--no-height_append",
    ])
    assert overridden.epochs == 2
    assert overridden.height_append is False


def test_macc_matches_a_hand_computed_confusion_matrix():
    """mAcc is per-class recall over the whole set, not a per-batch average."""
    train_mod = _load_train_module()

    class Stub:
        """Predicts class 0 for everything."""

        def predict_on_batch(self, points):
            out = np.zeros((len(points), 15), dtype=np.float32)
            out[:, 0] = 1.0
            return out

    # 10 objects of class 0, 5 of class 1: OA = 10/15; mAcc averages recall
    # over only the classes present -> (1.0 + 0.0) / 2 = 0.5.
    labels = np.array([0] * 10 + [1] * 5)
    points = np.zeros((15, 4, 3), dtype=np.float32)
    overall, macc = train_mod.evaluate(Stub(), points, labels, batch_size=4)

    assert overall == pytest.approx(10 / 15)
    assert macc == pytest.approx(0.5)


def _function_proxy(function):
    """Trace count, or compiled callable identity when Keras hides traces."""
    assert function is not None
    listing = getattr(function, "_list_all_concrete_functions", None)
    if listing is not None:
        return "traces", len(listing())
    return "callable", id(function)


def test_compiled_batch_functions_are_reused_across_epochs():
    """Repeated numpy epochs must not accumulate new traced functions."""
    train_mod = _load_train_module()
    tf = train_mod.tf
    model = tf.keras.Sequential(
        [tf.keras.layers.Input((4,)), tf.keras.layers.Dense(15)]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    points = np.zeros((9, 4), dtype=np.float32)
    labels = np.eye(15, dtype=np.float32)[np.arange(9) % 15]

    counts = []
    for _ in range(6):
        train_mod.train_one_epoch(model, points.copy(), labels.copy(), batch_size=4)
        train_mod.evaluate(model, points.copy(), np.arange(9) % 15, batch_size=4)
        counts.append(
            (
                _function_proxy(getattr(model, "train_function", None)),
                _function_proxy(getattr(model, "predict_function", None)),
            )
        )

    # Ignore the warm-up epoch: both cached functions must then remain fixed.
    assert all(count == counts[1] for count in counts[2:])
    assert int(model.optimizer.iterations.numpy()) == 6 * 3


def test_previous_metrics_are_preserved_with_incrementing_attempt_number(tmp_path):
    train_mod = _load_train_module()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"epochs_completed": 193, "per_epoch": []}))
    (tmp_path / "metrics.attempt1.json").write_text("older")

    archived, epochs = train_mod.preserve_previous_metrics(str(tmp_path))

    assert epochs == 193
    assert archived == str(tmp_path / "metrics.attempt2.json")
    assert json.loads((tmp_path / "metrics.attempt2.json").read_text())[
        "epochs_completed"
    ] == 193
    assert not metrics.exists()
