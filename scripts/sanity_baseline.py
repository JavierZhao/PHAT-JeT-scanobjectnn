#!/usr/bin/env python
"""Pipeline sanity check (handoff Task 1.4 / plan section 3.5).

Trains a minimal PointNet-style baseline on the real pipeline. This is a test
of the data path, not a result: if it cannot clear ~60% overall accuracy in a
few epochs, something upstream is wrong and nothing else should be launched.
"""

import argparse
import json
import os
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
    load_split,
    prepare_eval_sample,
    prepare_train_sample,
)


def _dense_bn_relu(x, units):
    x = tf.keras.layers.Dense(units, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def build_pointnet(num_points=NUM_POINTS, num_classes=NUM_CLASSES,
                   variant="pointnet"):
    """Build the gate model, or its former minimal variant for diagnostics.

    The gate intentionally omits PointNet's learned transform matrices, but
    retains its shared-MLP widths, batch normalization, max pooling, and
    classifier capacity.  This is still a fast pipeline check, not a reported
    benchmark model.
    """
    points = tf.keras.layers.Input((num_points, 3))
    x = points
    if variant == "minimal":
        x = tf.keras.layers.Dense(64, activation="relu")(points)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        logits = tf.keras.layers.Dense(num_classes)(x)
        return tf.keras.Model(points, logits, name="pointnet_sanity_minimal")
    if variant != "pointnet":
        raise ValueError(f"unknown PointNet variant: {variant!r}")

    for units in (64, 64, 64, 128, 1024):
        x = _dense_bn_relu(x, units)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = _dense_bn_relu(x, 512)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = _dense_bn_relu(x, 256)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(points, logits, name="pointnet_sanity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_points, train_labels = load_split(args.data_dir, "train")
    test_points, test_labels = load_split(args.data_dir, "test")
    print(f"train={train_points.shape} test={test_points.shape}", flush=True)

    subsample = eval_subsample_indices(test_points.shape[0], test_points.shape[1])
    test_prepared = np.stack(
        [prepare_eval_sample(test_points[i], subsample[i], "morton")
         for i in range(test_points.shape[0])]
    )

    model = build_pointnet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    started = time.time()
    best_oa = 0.0
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        order = rng.permutation(train_points.shape[0])
        batch = np.stack(
            [prepare_train_sample(train_points[i], rng, "morton") for i in order]
        )
        model.fit(batch, train_labels[order], batch_size=args.batch_size, epochs=1,
                  shuffle=False, verbose=0)

        preds = np.argmax(
            model.predict(test_prepared, batch_size=args.batch_size, verbose=0), axis=1
        )
        overall = float((preds == test_labels).mean())
        best_oa = max(best_oa, overall)
        print(f"epoch {epoch}: test_oa={overall:.4f}", flush=True)

    passed = best_oa >= args.threshold
    result = dict(
        best_test_oa=best_oa,
        threshold=args.threshold,
        passed=passed,
        epochs=args.epochs,
        wall_clock_s=round(time.time() - started, 1),
        tf_version=tf.__version__,
    )
    with open(os.path.join(args.out, "sanity.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    if not passed:
        print("SANITY CHECK FAILED -- do not launch the grid.", flush=True)
        sys.exit(1)
    print("SANITY CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
