#!/usr/bin/env python
"""Train PHAT-JeT on ScanObjectNN PB_T50_RS (plan section 3.4).

The historical recipe remains the default; ``--recipe pointnext`` selects the
PointNeXt training bundle, including raw y-height appending and no warmup.

Checkpoint selection uses a held-out split of the training set. The test set is
evaluated each epoch for the reported curves but never drives any decision.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.scanobjectnn import (  # noqa: E402
    NUM_CLASSES,
    NUM_POINTS,
    eval_subsample_indices,
    fixed_random_order,
    load_split,
    prepare_eval_sample,
    prepare_train_sample,
    stratified_train_val_split,
)
from models.phat_sonn import build_phat_sonn_classifier, count_flops  # noqa: E402


RECIPE_DEFAULTS = {
    "current": dict(epochs=250, batch_size=32, lr=1e-3, weight_decay=0.05,
                    warmup_epochs=10, label_smoothing=0.2,
                    height_append=False),
    "pointnext": dict(epochs=250, batch_size=32, lr=1e-3, weight_decay=0.05,
                      warmup_epochs=0, label_smoothing=0.2,
                      height_append=True, min_lr=1e-5, decay_epochs=None,
                      grad_clip=None),
    # Faithful to cfgs/scanobjectnn/default.yaml in the released PointNeXt
    # repo, which differs from the paper body in several places:
    #   lr 2e-3 (paper text says 1e-3 "unless otherwise specified"; the
    #     ScanObjectNN config is the otherwise)
    #   label_smoothing 0.32143 -- their epsilon is 0.3, but openpoints puts
    #     the true target at 1-eps while Keras puts it at (1-a)+a/K, so
    #     a = eps*K/(K-1) = 0.3*15/14 is what reproduces it at K=15
    #   cosine decays to min_lr 1e-4 over t_max=200, then holds constant for
    #     the final 50 epochs
    #   grad_norm_clip 10 (global norm, not per-variable)
    "pointnext_v2": dict(epochs=250, batch_size=32, lr=2e-3, weight_decay=0.05,
                         warmup_epochs=0, label_smoothing=0.32143,
                         height_append=True, min_lr=1e-4, decay_epochs=200,
                         grad_clip=10.0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", choices=["XS", "S", "M", "L"], default="S")
    p.add_argument("--delta", type=float, default=0.25, help="GMP voxel edge length")
    p.add_argument("--gmp", choices=["on", "off"], default="on")
    p.add_argument(
        "--downsample_stride", type=int, default=None,
        help="TRUE hierarchy: reduce the point count by this factor at the "
             "model midpoint, as the jet model's GeometricPooling did. Distinct "
             "from --hierarchy_pool_size, which keeps full resolution.",
    )
    p.add_argument(
        "--delta_growth", choices=["density", "double", "none"], default="density",
        help="How the GMP voxel grows after downsampling. 'none' isolates "
             "hierarchy from receptive-field scaling.",
    )
    p.add_argument(
        "--gmp_kernel", type=int, default=3,
        help="GMP convolution kernel size -- the GMP spatial extent, "
             "independent of delta (which sets resolution).",
    )
    p.add_argument(
        "--gmp_variant",
        choices=["dense", "sparse", "sparse_mean", "sparse_trilinear"],
        default="dense",
        help="GMP implementation. 'dense' is the legacy default that all "
             "completed runs used; the sparse variants cost roughly constant "
             "time in grid resolution and make very fine delta affordable.",
    )
    p.add_argument("--ordering", choices=["morton", "random"], default="morton")
    p.add_argument(
        "--patch_size", type=int, default=None,
        help="override the config's patch size (Phase D)",
    )
    p.add_argument(
        "--shifted_patches", action="store_true",
        help="shift odd PHAT blocks by half a patch to overlap neighbourhoods",
    )
    p.add_argument(
        "--hierarchy_pool_size", type=int, default=None,
        help="enable one coarse PHAT path by pooling this many adjacent points",
    )
    p.add_argument(
        "--hierarchy_after_block", type=int, default=None,
        help="zero-based block after which to insert the coarse path (default: midpoint)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recipe", choices=RECIPE_DEFAULTS, default="current")
    p.add_argument(
        "--height_mode", choices=["raw", "shifted", "unit", "xyz_shifted"], default="raw",
        help="Appended-height definition: raw centred height, shifted above "
             "the object base, or unit (normalized y -- a control that adds "
             "no information beyond xyz).",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument(
        "--height_append", action=argparse.BooleanOptionalAction, default=None,
        help="append raw y height as a fourth per-point input feature",
    )
    p.add_argument("--local_attention", choices=["on", "off"], default="on",
                   help="ablate the within-patch attention. There has never "
                        "been an attention ablation -- only a GMP one.")
    p.add_argument("--patch_messages", choices=["on", "off"], default="on",
                   help="ablate the patch-token global path.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout inside PHAT blocks. Never exposed before, so "
                        "every run to date used 0.0 -- the model has had no "
                        "regularization beyond weight decay and label smoothing.")
    p.add_argument("--pool", choices=["mean", "max", "maxmean"], default="mean",
                   help="global readout. The jet model used max; the 3D port "
                        "defaulted to mean.")
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--head_width", type=int, default=None,
                   help="head hidden width; defaults to d_model//2")
    p.add_argument("--final_norm", action="store_true",
                   help="LayerNorm before the head (pre-LN trunks need it)")
    p.add_argument("--min_lr", type=float, default=None,
                   help="cosine floor; the schedule holds here after decay_epochs")
    p.add_argument("--decay_epochs", type=int, default=None,
                   help="cosine decay length; defaults to the full run")
    p.add_argument("--grad_clip", type=float, default=None,
                   help="global gradient-norm clip (Keras global_clipnorm)")
    p.add_argument("--val_fraction", type=float, default=0.1)
    args = p.parse_args(argv)
    for name, value in RECIPE_DEFAULTS[args.recipe].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def git_sha():
    env = os.environ.get("GIT_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def make_train_epoch(points, labels, args, fixed_perm, epoch):
    """Materialize one epoch of resampled, augmented, ordered clouds.

    Built eagerly as numpy rather than through a tf.data generator: at this
    dataset size (~140 MB per epoch, a few seconds to build) it costs little
    next to an epoch of GPU time, and it keeps augmentation explicit,
    reproducible and free of tf.data's threading behaviour, which differs
    between the Keras 2 and Keras 3 runtimes.
    """
    rng = np.random.default_rng(args.seed * 100_000 + epoch)
    order = rng.permutation(points.shape[0])
    batch = np.stack(
        [prepare_train_sample(points[i], rng, args.ordering, fixed_perm,
                              height_append=args.height_append,
                              height_mode=args.height_mode)
         for i in order]
    )
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[labels[order]]
    return batch, one_hot


def make_eval_arrays(points, labels, args, fixed_perm):
    """Deterministic: fixed subsample indices, no augmentation, no voting."""
    if points.shape[0] == 0:
        # --val_fraction 0 leaves no validation set at all.
        width = 6 if args.height_mode == "xyz_shifted" else (
            4 if args.height_append else 3)
        return np.empty((0, NUM_POINTS, width), np.float32), labels
    subsample = eval_subsample_indices(points.shape[0], points.shape[1])
    prepared = np.stack(
        [
            prepare_eval_sample(points[i], subsample[i], args.ordering, fixed_perm,
                                height_append=args.height_append,
                              height_mode=args.height_mode)
            for i in range(points.shape[0])
        ]
    )
    return prepared, labels


def evaluate(model, points, labels, batch_size):
    """Overall accuracy and mean class accuracy from a full confusion matrix."""
    # Batch explicitly instead of calling predict() on a new numpy array each
    # epoch.  predict() constructs a fresh data adapter/iterator per call; on
    # TF/Keras 2.13 those objects retain host buffers over long runs.  The
    # compiled predict_on_batch function is cached and reused.
    predicted = []
    for start in range(0, len(points), batch_size):
        predicted.append(model.predict_on_batch(points[start:start + batch_size]))
    preds = np.argmax(np.concatenate(predicted, axis=0), axis=1)
    overall = float((preds == labels).mean())

    per_class = []
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        # Every class is present in the ScanObjectNN test split; load_split
        # asserts this, so mAcc is always defined over all 15 classes.
        if mask.sum():
            per_class.append(float((preds[mask] == cls).mean()))
    return overall, float(np.mean(per_class))


def train_one_epoch(model, points, one_hot, batch_size):
    """Train on one already-shuffled numpy epoch using one compiled function.

    This has the same batch boundaries and optimizer updates as fit(...,
    epochs=1, shuffle=False), without constructing a new numpy data adapter on
    every epoch.  Keras' loss metric is reset once here, as fit does at the
    beginning of an epoch, so the final returned loss is the epoch aggregate.
    """
    model.reset_metrics()
    result = None
    for start in range(0, len(points), batch_size):
        result = model.train_on_batch(
            points[start:start + batch_size],
            one_hot[start:start + batch_size],
        )
    if result is None:
        raise ValueError("cannot train on an empty epoch")
    if isinstance(result, dict):
        return float(result["loss"])
    if isinstance(result, (list, tuple)):
        return float(result[0])
    return float(result)


def preserve_previous_metrics(out_dir):
    """Move an earlier attempt's metrics aside before starting from epoch 0."""
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    epochs_completed = "unknown"
    try:
        with open(metrics_path) as handle:
            epochs_completed = json.load(handle).get("epochs_completed", "unknown")
    except (OSError, ValueError, TypeError):
        pass

    attempt = 1
    while os.path.exists(os.path.join(out_dir, f"metrics.attempt{attempt}.json")):
        attempt += 1
    archived = os.path.join(out_dir, f"metrics.attempt{attempt}.json")
    os.replace(metrics_path, archived)
    return archived, epochs_completed


def build_classifier(args):
    """Build the stock 3D model, or expose height only to its input embedding.

    Existing PHAT blocks receive xyz coordinates exactly as before, so GMP
    voxel indices remain three-dimensional.  The only widened layer is the
    intended per-point input embedding.
    """
    base = build_phat_sonn_classifier(
        config=args.config,
        num_points=NUM_POINTS,
        num_classes=NUM_CLASSES,
        grid_size=args.delta,
        use_gmp=(args.gmp == "on"),
        gmp_variant=args.gmp_variant,
        patch_size=args.patch_size,
        shifted_patches=args.shifted_patches,
        hierarchy_pool_size=args.hierarchy_pool_size,
        downsample_stride=args.downsample_stride,
        delta_growth=args.delta_growth,
        gmp_kernel=args.gmp_kernel,
        dropout=args.dropout,
        pool=args.pool,
        use_local_attention=(args.local_attention == "on"),
        use_patch_messages=(args.patch_messages == "on"),
        head_dropout=args.head_dropout,
        head_width=args.head_width,
        final_norm=args.final_norm,
        hierarchy_after_block=args.hierarchy_after_block,
    )
    if not args.height_append:
        return base

    # xyz_shifted appends three absolute-offset channels, the others one.
    extra = 3 if args.height_mode == "xyz_shifted" else 1
    features = tf.keras.layers.Input((NUM_POINTS, 3 + extra), name="points")
    coords = tf.keras.layers.Lambda(
        lambda tensor: tensor[..., :3], name="xyz_coordinates"
    )(features)
    embedding = base.get_layer("input_embedding")
    x = tf.keras.layers.Dense(embedding.units, name="input_embedding")(features)
    for layer in base.layers:
        if layer.name.startswith("phat_block3d_"):
            x, coords = layer([x, coords])
        elif layer.name == "coarse_phat_path":
            x, coords = layer([x, coords])
    # Rebuild pooling rather than reusing it by name. It carries no weights,
    # and for pool="maxmean" it is two layers feeding a Concatenate, so it is
    # not reusable as a single callable -- that combination crashed on the
    # cluster.
    if any(layer.name == "final_norm" for layer in base.layers):
        x = base.get_layer("final_norm")(x)
    if args.pool == "max":
        x = tf.keras.layers.GlobalMaxPooling1D(name="global_pool")(x)
    elif args.pool == "maxmean":
        x = tf.keras.layers.Concatenate(name="global_pool")(
            [tf.keras.layers.GlobalMaxPooling1D()(x),
             tf.keras.layers.GlobalAveragePooling1D()(x)]
        )
    else:
        x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
    x = base.get_layer("head_hidden")(x)
    for name in ("head_dropout", "dropout"):
        if any(layer.name == name for layer in base.layers):
            x = base.get_layer(name)(x)
    logits = base.get_layer("logits")(x)
    return tf.keras.Model(features, logits, name=base.name)


def model_for_flops(model, height_append, height_mode="raw"):
    """Give the legacy three-channel FLOPs helper a shape-compatible graph."""
    if not height_append:
        return model
    xyz = tf.keras.layers.Input((NUM_POINTS, 3), name="flops_xyz")
    extra = xyz if height_mode == "xyz_shifted" else xyz[..., 1:2]
    inputs = tf.keras.layers.Concatenate(axis=-1)([xyz, extra])
    return tf.keras.Model(xyz, model(inputs))


class MetricsLog:
    """Per-epoch val/test evaluation and an always-valid metrics.json."""

    def __init__(self, model, val, test, header, out_dir, batch_size):
        self.model = model
        self.val_points, self.val_labels = val
        self.test_points, self.test_labels = test
        self.header = header
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.records = []
        self.best_val_oa = -1.0
        self.started = time.time()

    def record_epoch(self, epoch, train_loss):
        if len(self.val_labels):
            val_oa, val_macc = evaluate(
                self.model, self.val_points, self.val_labels, self.batch_size
            )
        else:
            val_oa, val_macc = float("nan"), float("nan")
        test_oa, test_macc = evaluate(
            self.model, self.test_points, self.test_labels, self.batch_size
        )

        # Checkpoint selection is on validation only; the test curve is
        # recorded but never drives a decision. With no val split there is no
        # selection at all -- the final epoch is the result.
        if len(self.val_labels) and val_oa > self.best_val_oa:
            self.best_val_oa = val_oa
            self.model.save_weights(os.path.join(self.out_dir, "best_val.weights.h5"))

        self.records.append(
            dict(
                epoch=int(epoch),
                train_loss=float(train_loss),
                val_oa=val_oa,
                val_macc=val_macc,
                test_oa=test_oa,
                test_macc=test_macc,
                elapsed_s=round(time.time() - self.started, 1),
            )
        )
        self.write()
        logging.info(
            "epoch %d loss=%.4f val_oa=%.4f test_oa=%.4f test_macc=%.4f",
            epoch, train_loss, val_oa, test_oa, test_macc,
        )

    def write(self, extra=None):
        payload = dict(self.header)
        payload["epochs_completed"] = len(self.records)
        payload["per_epoch"] = self.records
        if extra:
            payload.update(extra)
        # Write-then-rename: a killed job never leaves a truncated file.
        tmp = os.path.join(self.out_dir, "metrics.json.tmp")
        with open(tmp, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, os.path.join(self.out_dir, "metrics.json"))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    previous_metrics = preserve_previous_metrics(args.out)
    logging.basicConfig(
        filename=os.path.join(args.out, "train.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if previous_metrics is not None:
        archived, epochs_completed = previous_metrics
        logging.warning(
            "FOUND PREVIOUS ATTEMPT with %s completed epochs; preserved it as %s "
            "before restarting from epoch 0",
            epochs_completed, archived,
        )
    set_seeds(args.seed)
    logging.info("args: %s", vars(args))

    train_points, train_labels = load_split(args.data_dir, "train")
    test_points, test_labels = load_split(args.data_dir, "test")
    if args.val_fraction > 0:
        train_idx, val_idx = stratified_train_val_split(
            train_labels, val_fraction=args.val_fraction
        )
    else:
        # Train on all 11,416 objects, as PointNeXt and PointMLP do. With no
        # held-out split there is nothing to select a checkpoint on, so the
        # honest report is the FINAL epoch; test_curve_max is also recorded
        # because that is what the published baselines effectively report.
        train_idx = np.arange(len(train_labels))
        val_idx = np.empty(0, dtype=int)
    logging.info("train=%d val=%d test=%d", len(train_idx), len(val_idx),
                 len(test_labels))

    fixed_perm = fixed_random_order() if args.ordering == "random" else None
    epoch_points = train_points[train_idx]
    epoch_labels = train_labels[train_idx]
    val = make_eval_arrays(
        train_points[val_idx], train_labels[val_idx], args, fixed_perm
    )
    test = make_eval_arrays(test_points, test_labels, args, fixed_perm)

    model = build_classifier(args)

    steps_per_epoch = int(np.ceil(len(train_idx) / args.batch_size))
    # Keras CosineDecay clamps to alpha*initial past decay_steps, so a
    # decay_epochs shorter than epochs reproduces PointNeXt's constant-LR
    # anneal-out over the final stretch.
    decay_epochs = args.decay_epochs or args.epochs
    min_lr = args.min_lr if args.min_lr is not None else 1e-5
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=max(1, (decay_epochs - args.warmup_epochs) * steps_per_epoch),
        alpha=min_lr / args.lr,
        warmup_target=args.lr,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
    )
    optimizer_kwargs = dict(learning_rate=schedule, weight_decay=args.weight_decay)
    if args.grad_clip:
        # global_clipnorm clips the whole gradient vector, matching torch's
        # clip_grad_norm_. Keras `clipnorm` is per-variable and is NOT the same.
        optimizer_kwargs["global_clipnorm"] = args.grad_clip
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(**optimizer_kwargs),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=args.label_smoothing
        ),
    )

    header = dict(
        args=vars(args),
        git_sha=git_sha(),
        image_digest=os.environ.get("IMAGE_DIGEST", "unknown"),
        tf_version=tf.__version__,
        params=int(model.count_params()),
        num_train=len(train_idx),
        num_val=len(val_idx),
        num_test=int(len(test_labels)),
    )
    logging.info("header: %s", header)

    metrics = MetricsLog(model, val, test, header, args.out, args.batch_size)
    for epoch in range(args.epochs):
        points, one_hot = make_train_epoch(
            epoch_points, epoch_labels, args, fixed_perm, epoch
        )
        # Reuse the compiled batch function.  The cosine schedule advances
        # with optimizer.iterations exactly once per batch across epochs.
        train_loss = train_one_epoch(model, points, one_hot, args.batch_size)
        metrics.record_epoch(epoch, train_loss)
        del points, one_hot

    model.save_weights(os.path.join(args.out, "final.weights.h5"))
    final_test_oa, final_test_macc = evaluate(model, test[0], test[1], args.batch_size)

    # Test numbers at the best-validation epoch: the headline result when a
    # val split exists. With --val_fraction 0 there is no such epoch, and the
    # final-epoch number above is the honest one.
    best_ckpt = os.path.join(args.out, "best_val.weights.h5")
    if len(val[1]) and os.path.exists(best_ckpt):
        model.load_weights(best_ckpt)
        best_test_oa, best_test_macc = evaluate(model, test[0], test[1], args.batch_size)
    else:
        best_test_oa = best_test_macc = None

    # Profiled last: it builds a separate graph and is version-sensitive, so a
    # failure here must not cost a completed training run.
    try:
        flops = int(count_flops(model_for_flops(model, args.height_append, args.height_mode), NUM_POINTS))
    except Exception as exc:
        logging.warning("FLOPs profiling failed: %s", exc)
        flops = None

    metrics.write(
        extra=dict(
            flops=flops,
            final=dict(test_oa=final_test_oa, test_macc=final_test_macc),
            at_best_val=dict(test_oa=best_test_oa, test_macc=best_test_macc),
            # Descriptive only -- selecting on this would be tuning on test.
            test_curve_max=dict(
                test_oa=max(r["test_oa"] for r in metrics.records),
                test_macc=max(r["test_macc"] for r in metrics.records),
            ),
            wall_clock_s=round(time.time() - metrics.started, 1),
            completed=True,
        )
    )
    logging.info(
        "done: at_best_val OA=%.4f mAcc=%.4f | final OA=%.4f mAcc=%.4f",
        best_test_oa, best_test_macc, final_test_oa, final_test_macc,
    )


if __name__ == "__main__":
    main()
