"""Tests for true hierarchical downsampling and receptive-field scaling."""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hierarchy import GeometricPooling3D, stage_grid_size


def _inputs(batch=2, num_points=64, channels=8, seed=0):
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.normal(size=(batch, num_points, channels)).astype(np.float32))
    coords = tf.constant(
        rng.uniform(-1.0, 1.0, size=(batch, num_points, 3)).astype(np.float32)
    )
    return x, coords


@pytest.mark.parametrize("stride", [2, 4])
def test_pooling_reduces_point_count_and_projects(stride):
    x, coords = _inputs()
    pool = GeometricPooling3D(out_dim=16, stride=stride)
    out_x, out_coords = pool([x, coords])
    assert out_x.shape == (2, 64 // stride, 16)
    assert out_coords.shape == (2, 64 // stride, 3)


def test_pooled_coordinates_are_group_centroids():
    """Each surviving point must sit at the centroid of the group it summarizes."""
    x, coords = _inputs(num_points=8, seed=1)
    pool = GeometricPooling3D(out_dim=4, stride=2)
    _, out_coords = pool([x, coords])
    expected = coords.numpy().reshape(2, 4, 2, 3).mean(axis=2)
    np.testing.assert_allclose(out_coords.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_features_are_max_pooled_like_the_jet_model():
    x, coords = _inputs(num_points=8, channels=4, seed=2)
    pool = GeometricPooling3D(out_dim=4, stride=2)
    pool([x, coords])  # build
    pooled = x.numpy().reshape(2, 4, 2, 4).max(axis=2)
    expected = pool.norm(pool.proj(tf.constant(pooled))).numpy()
    out_x, _ = pool([x, coords])
    np.testing.assert_allclose(out_x.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_pooling_rejects_indivisible_point_counts():
    x, coords = _inputs(num_points=7)
    with pytest.raises(ValueError, match="divisible"):
        GeometricPooling3D(out_dim=4, stride=2)([x, coords])


def test_pooling_rejects_stride_of_one():
    with pytest.raises(ValueError, match="stride"):
        GeometricPooling3D(out_dim=4, stride=1)


def test_gradients_flow_through_pooling():
    x, coords = _inputs(seed=3)
    pool = GeometricPooling3D(out_dim=16, stride=2)
    with tf.GradientTape() as tape:
        tape.watch(x)
        out_x, _ = pool([x, coords])
        loss = tf.reduce_mean(tf.square(out_x))
    grad_in, grad_w = tape.gradient(loss, [x, pool.proj.kernel])
    assert grad_in is not None and np.abs(grad_in.numpy()).sum() > 0
    assert grad_w is not None and np.abs(grad_w.numpy()).sum() > 0


def test_pooling_runs_under_tf_function():
    x, coords = _inputs(seed=4)
    pool = GeometricPooling3D(out_dim=16, stride=2)

    @tf.function
    def forward(a, b):
        return pool([a, b])

    eager = pool([x, coords])[0].numpy()
    traced = forward(x, coords)[0].numpy()
    np.testing.assert_allclose(traced, eager, rtol=1e-5, atol=1e-6)


def test_morton_contiguous_groups_stay_spatially_local():
    """Pooling consecutive points is only sensible if they are near each other.

    This is what lets us skip the jet model's explicit sort: the data pipeline
    already delivers Morton order.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from data.scanobjectnn import morton_order, normalize_unit_sphere

    rng = np.random.default_rng(5)
    cloud = normalize_unit_sphere(rng.normal(size=(1024, 3)).astype(np.float32))
    ordered = cloud[morton_order(cloud)]

    pair_spread = np.linalg.norm(
        ordered.reshape(512, 2, 3) - ordered.reshape(512, 2, 3).mean(axis=1, keepdims=True),
        axis=-1,
    ).mean()
    shuffled = cloud[rng.permutation(1024)]
    random_spread = np.linalg.norm(
        shuffled.reshape(512, 2, 3) - shuffled.reshape(512, 2, 3).mean(axis=1, keepdims=True),
        axis=-1,
    ).mean()
    assert pair_spread < random_spread / 2, (
        f"Morton pairs spread {pair_spread:.4f} vs random {random_spread:.4f}"
    )


def test_density_growth_keeps_points_per_voxel_roughly_constant():
    """Halving the point count should grow voxel volume by ~2x, edge by 2**(1/3)."""
    base = 0.09375
    grown = stage_grid_size(base, stage=1, stride=2, growth="density")
    np.testing.assert_allclose(grown / base, 2 ** (1 / 3), rtol=1e-6)
    # volume ratio is the quantity that matters
    np.testing.assert_allclose((grown / base) ** 3, 2.0, rtol=1e-6)


def test_growth_modes():
    base = 0.09375
    assert stage_grid_size(base, 0, 2, "density") == base
    assert stage_grid_size(base, 2, 2, "none") == base
    np.testing.assert_allclose(stage_grid_size(base, 2, 2, "double"), base * 4)
    with pytest.raises(ValueError):
        stage_grid_size(base, 1, 2, "bogus")
