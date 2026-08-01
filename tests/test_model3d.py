"""End-to-end model tests for all four configs (plan section 3.3)."""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.scanobjectnn import NUM_CLASSES, NUM_POINTS
from models.phat_sonn import CONFIGS, build_phat_sonn_classifier

# Nominal targets from the handoff. The ladder is kept as specified and the
# real counts are reported as measured (plan section 1.2); this test pins the
# measured values so an unintended architecture change is caught.
MEASURED_PARAMS = {"XS": 156_559, "S": 612_111, "M": 1_214_479, "L": 2_712_591}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_builds_and_forwards(config):
    model = build_phat_sonn_classifier(config=config)
    out = model(tf.zeros([2, NUM_POINTS, 3]), training=False)
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_parameter_count_is_unchanged(config):
    """Guards against silent architecture drift."""
    model = build_phat_sonn_classifier(config=config)
    assert model.count_params() == MEASURED_PARAMS[config]


def test_gmp_off_removes_only_the_gmp_parameters():
    on = build_phat_sonn_classifier(config="S", use_gmp=True)
    off = build_phat_sonn_classifier(config="S", use_gmp=False)
    assert off.count_params() < on.count_params()
    out = off(tf.zeros([2, NUM_POINTS, 3]), training=False)
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("patch_size", [16, 32, 64, 128])
def test_patch_size_override(patch_size):
    """Phase D sweeps P on config M."""
    model = build_phat_sonn_classifier(config="M", patch_size=patch_size)
    out = model(tf.zeros([2, NUM_POINTS, 3]), training=False)
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("grid_size", [0.5, 0.25, 0.125])
def test_delta_anchors_all_run(grid_size):
    model = build_phat_sonn_classifier(config="S", grid_size=grid_size)
    points = tf.constant(
        np.random.default_rng(0)
        .uniform(-1.0, 1.0, size=(2, NUM_POINTS, 3))
        .astype(np.float32)
    )
    assert model(points, training=False).shape == (2, NUM_CLASSES)


def test_output_is_logits_not_probabilities():
    """The loss applies softmax with from_logits=True."""
    model = build_phat_sonn_classifier(config="XS")
    points = tf.constant(
        np.random.default_rng(1)
        .normal(size=(8, NUM_POINTS, 3))
        .astype(np.float32)
    )
    out = model(points, training=False).numpy()
    sums = out.sum(axis=-1)
    assert not np.allclose(sums, 1.0, atol=1e-3), "head must not apply softmax"


def test_one_training_step_reduces_loss_on_a_fixed_batch():
    """Loss contract check: one-hot labels + label smoothing on logits."""
    rng = np.random.default_rng(2)
    points = tf.constant(rng.normal(size=(8, NUM_POINTS, 3)).astype(np.float32))
    labels = tf.constant(
        tf.keras.utils.to_categorical(rng.integers(0, NUM_CLASSES, 8), NUM_CLASSES)
    )

    model = build_phat_sonn_classifier(config="XS")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.2
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-3)

    losses = []
    for _ in range(6):
        with tf.GradientTape() as tape:
            loss = loss_fn(labels, model(points, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_gradients_reach_the_gmp_conv():
    """The positional prior must actually train."""
    rng = np.random.default_rng(3)
    points = tf.constant(rng.normal(size=(4, NUM_POINTS, 3)).astype(np.float32))
    model = build_phat_sonn_classifier(config="XS")

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(points, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)

    # Keras 3 exposes Variable.path; Keras 2 (TF 2.13 on the cluster) only has
    # .name. The suite must pass under both.
    def var_id(variable):
        return (getattr(variable, "path", None) or variable.name).lower()

    gmp_grads = [
        g for g, v in zip(grads, model.trainable_variables)
        if "conv3d" in var_id(v) and g is not None
    ]
    assert gmp_grads, "no Conv3D variables found in the model"
    assert any(np.abs(g.numpy()).sum() > 0 for g in gmp_grads)


@pytest.mark.parametrize(
    "gmp_variant", ["dense", "sparse", "sparse_mean", "sparse_trilinear"]
)
def test_gmp_variant_is_threaded_through_classifier(gmp_variant):
    model = build_phat_sonn_classifier(
        config="XS", num_points=64, gmp_variant=gmp_variant, grid_size=0.125
    )
    out = model(tf.zeros([2, 64, 3]), training=False)
    assert out.shape == (2, NUM_CLASSES)
