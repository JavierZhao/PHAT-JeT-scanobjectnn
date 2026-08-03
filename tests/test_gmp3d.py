"""Unit tests for 3D GMP (plan section 3.2). CPU-only; no dataset required."""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gmp3d import GeometricMessagePassing3D, quantize, scatter_to_grid


SPARSE_VARIANTS = ("sparse", "sparse_mean", "sparse_trilinear")


def _cloud(batch=2, num_points=64, channels=8, seed=0, scales=(1.0, 0.6)):
    """Random unit-sphere-ish clouds with deliberately unequal per-cloud extents."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.0, 1.0, size=(batch, num_points, 3)).astype(np.float32)
    for b in range(batch):
        coords[b] *= scales[b % len(scales)]
    feats = rng.normal(size=(batch, num_points, channels)).astype(np.float32)
    return tf.constant(coords), tf.constant(feats)


def test_shape_is_preserved():
    """N is preserved -- the defining property vs. tokenization."""
    coords, feats = _cloud()
    layer = GeometricMessagePassing3D(channels=8, kernel_size=3, grid_size=0.25)
    out = layer(feats, coords)
    assert out.shape == feats.shape


def test_duplicate_points_in_a_voxel_are_summed():
    """Two points in one cell must add, not overwrite (scatter_nd semantics)."""
    coords = tf.constant([[[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]]], dtype=tf.float32)
    feats = tf.constant([[[1.0, 2.0], [10.0, 20.0]]], dtype=tf.float32)

    idx = quantize(coords, grid_size=0.5)
    np.testing.assert_array_equal(idx.numpy()[0, 0], idx.numpy()[0, 1])

    grid = scatter_to_grid(feats, idx, [1, 1, 1])
    np.testing.assert_allclose(grid.numpy().reshape(-1), [11.0, 22.0], rtol=1e-6)


def test_quantize_is_non_negative_and_min_shifted():
    coords, _ = _cloud()
    idx = quantize(coords, grid_size=0.25).numpy()
    assert idx.min() == 0, "per-cloud min-shift must put the minimum at index 0"
    assert (idx >= 0).all()


@pytest.mark.parametrize("grid_size", [0.5, 0.25, 0.125])
def test_permutation_equivariance(grid_size):
    """Permuting the input points permutes the output identically.

    Tolerance-based: duplicate-voxel accumulation order can differ under a
    permutation, so bitwise equality is too strong a claim.
    """
    coords, feats = _cloud(seed=1)
    layer = GeometricMessagePassing3D(channels=8, kernel_size=3, grid_size=grid_size)
    out = layer(feats, coords).numpy()

    rng = np.random.default_rng(7)
    perm = rng.permutation(coords.shape[1])
    out_perm = layer(
        tf.gather(feats, perm, axis=1), tf.gather(coords, perm, axis=1)
    ).numpy()

    np.testing.assert_allclose(out_perm, out[:, perm, :], rtol=1e-4, atol=1e-5)


def test_gradients_flow_to_input_and_conv_kernel():
    """Squared-error loss against a random target: non-degenerate through LN."""
    coords, feats = _cloud(seed=2)
    layer = GeometricMessagePassing3D(channels=8, kernel_size=3, grid_size=0.25)
    target = tf.constant(
        np.random.default_rng(3).normal(size=feats.shape).astype(np.float32)
    )

    with tf.GradientTape() as tape:
        tape.watch(feats)
        out = layer(feats, coords)
        loss = tf.reduce_mean(tf.square(out - target))

    grad_in, grad_kernel = tape.gradient(
        loss, [feats, layer.conv3d.kernel]
    )
    assert grad_in is not None and np.abs(grad_in.numpy()).sum() > 0
    assert grad_kernel is not None and np.abs(grad_kernel.numpy()).sum() > 0, (
        "no gradient reached the depthwise conv -- scatter/gather path is broken"
    )


def test_runs_under_tf_function():
    """The layer must trace: training runs inside a compiled graph."""
    coords, feats = _cloud(seed=4)
    layer = GeometricMessagePassing3D(channels=8, kernel_size=3, grid_size=0.25)

    @tf.function
    def forward(f, c):
        return layer(f, c)

    eager = layer(feats, coords).numpy()
    traced = forward(feats, coords).numpy()
    np.testing.assert_allclose(traced, eager, rtol=1e-5, atol=1e-6)


def test_grid_side_stays_within_memory_budget():
    """delta anchors must keep the grid side <= 32 (plan section 3.2)."""
    rng = np.random.default_rng(5)
    # Worst case: unit sphere inflated by the 1.1 scale augmentation.
    coords = tf.constant(
        rng.uniform(-1.1, 1.1, size=(4, 1024, 3)).astype(np.float32)
    )
    for grid_size, expected_max in [(0.5, 32), (0.25, 32), (0.125, 32)]:
        idx = quantize(coords, grid_size).numpy()
        assert idx.max() + 1 <= expected_max, (
            f"delta={grid_size} gives grid side {idx.max() + 1} > {expected_max}"
        )


@pytest.mark.parametrize("variant", SPARSE_VARIANTS)
def test_sparse_variants_preserve_shape_and_run_under_tf_function(variant):
    coords, feats = _cloud(num_points=24, channels=4, seed=10)
    layer = GeometricMessagePassing3D(
        channels=4, kernel_size=3, grid_size=0.125, variant=variant
    )

    @tf.function
    def forward(f, c):
        return layer(f, c)

    eager = layer(feats, coords)
    traced = forward(feats, coords)
    assert eager.shape == feats.shape
    np.testing.assert_allclose(traced.numpy(), eager.numpy(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("variant", SPARSE_VARIANTS)
def test_sparse_variants_are_permutation_equivariant(variant):
    coords, feats = _cloud(num_points=32, channels=4, seed=11)
    layer = GeometricMessagePassing3D(
        channels=4, kernel_size=3, grid_size=0.2, variant=variant
    )
    out = layer(feats, coords).numpy()
    perm = np.random.default_rng(12).permutation(coords.shape[1])
    permuted = layer(
        tf.gather(feats, perm, axis=1), tf.gather(coords, perm, axis=1)
    ).numpy()
    np.testing.assert_allclose(permuted, out[:, perm], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("variant", SPARSE_VARIANTS)
def test_sparse_variant_gradients_reach_input_and_kernel(variant):
    coords, feats = _cloud(num_points=24, channels=4, seed=13)
    layer = GeometricMessagePassing3D(
        channels=4, kernel_size=3, grid_size=0.25, variant=variant
    )
    with tf.GradientTape() as tape:
        tape.watch(feats)
        loss = tf.reduce_sum(tf.square(layer(feats, coords)))
    grad_input, grad_kernel = tape.gradient(loss, [feats, layer.conv3d.kernel])
    assert grad_input is not None and np.abs(grad_input.numpy()).sum() > 0
    assert grad_kernel is not None and np.abs(grad_kernel.numpy()).sum() > 0


def test_sparse_is_numerically_equivalent_to_dense_with_shared_weights():
    """Sparse lookup computes the same grouped Conv3D at every point voxel."""
    coords, feats = _cloud(batch=2, num_points=40, channels=4, seed=14)
    dense = GeometricMessagePassing3D(
        channels=4, kernel_size=3, grid_size=0.2, variant="dense"
    )
    sparse = GeometricMessagePassing3D(
        channels=4, kernel_size=3, grid_size=0.2, variant="sparse"
    )
    dense_out = dense(feats, coords)
    sparse(feats, coords)  # build all weights before copying
    sparse.set_weights(dense.get_weights())
    sparse_out = sparse(feats, coords)
    np.testing.assert_allclose(
        sparse_out.numpy(), dense_out.numpy(), rtol=2e-5, atol=2e-6
    )


def test_single_scale_bank_is_identical_to_no_bank():
    """A one-element gmp_scales must reproduce the original model exactly.

    This is what makes the option safe to add: ~90 completed runs depend on
    the single-scale path being untouched.
    """
    import numpy as np
    from models.phat_sonn import build_phat_sonn_classifier as build

    plain = build(config="XS", num_points=64, grid_size=0.09375)
    banked = build(config="XS", num_points=64, grid_size=0.09375,
                   gmp_scales=[0.09375])
    assert plain.count_params() == banked.count_params()

    points = tf.constant(
        np.random.default_rng(0).uniform(-1, 1, size=(2, 64, 3)).astype(np.float32)
    )
    banked.set_weights(plain.get_weights())
    np.testing.assert_allclose(
        banked(points, training=False).numpy(),
        plain(points, training=False).numpy(), rtol=1e-5, atol=1e-6,
    )


def test_multi_scale_bank_builds_every_requested_resolution():
    from models.phat_sonn import build_phat_sonn_classifier as build

    scales = [0.0625, 0.09375, 0.125]
    model = build(config="XS", num_points=64, gmp_scales=scales)
    model(tf.zeros([2, 64, 3]), training=False)
    block = [l for l in model.layers if "phat_block3d" in l.name][0]
    assert [g.grid_size for g in block.gmp] == scales
    assert model.count_params() > build(config="XS", num_points=64).count_params()


def test_multi_scale_gradients_reach_every_scale():
    """Each scale must train, or the bank is decoration."""
    import numpy as np
    from models.phat_sonn import build_phat_sonn_classifier as build

    model = build(config="XS", num_points=64, gmp_scales=[0.0625, 0.125])
    points = tf.constant(
        np.random.default_rng(1).uniform(-1, 1, size=(2, 64, 3)).astype(np.float32)
    )
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(points, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)

    def var_id(v):
        return (getattr(v, "path", None) or v.name).lower()

    for scale in ("gmp_scale_0", "gmp_scale_1"):
        touched = [
            g for g, v in zip(grads, model.trainable_variables)
            if scale in var_id(v) and g is not None and np.abs(g.numpy()).sum() > 0
        ]
        assert touched, f"no gradient reached {scale}"
