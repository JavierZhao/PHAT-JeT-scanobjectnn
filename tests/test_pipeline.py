"""Data-pipeline tests (plan section 3.1/3.3). Synthetic data; no dataset needed."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.scanobjectnn import (
    NUM_CLASSES,
    NUM_POINTS,
    apply_order,
    eval_subsample_indices,
    fixed_random_order,
    morton_order,
    normalize_unit_sphere,
    prepare_eval_sample,
    prepare_train_sample,
    random_rotation_z,
    random_scale,
    stratified_train_val_split,
)


def _raw(num=2048, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 5.0, size=(num, 3)).astype(np.float32)


def test_normalize_gives_zero_mean_unit_radius():
    points = normalize_unit_sphere(_raw())
    np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(points, axis=-1).max(), 1.0, atol=1e-5)


def test_augmentation_stays_within_the_delta_memory_assumption():
    """Scale <=1.1 on a unit sphere keeps coords in [-1.1, 1.1]."""
    rng = np.random.default_rng(1)
    points = normalize_unit_sphere(_raw())
    for _ in range(50):
        augmented = random_rotation_z(random_scale(points, rng), rng)
        assert np.abs(augmented).max() <= 1.1 + 1e-5


def test_rotation_preserves_shape_and_z():
    rng = np.random.default_rng(2)
    points = normalize_unit_sphere(_raw())
    rotated = random_rotation_z(points, rng)
    assert rotated.shape == points.shape
    np.testing.assert_allclose(rotated[:, 2], points[:, 2], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(points, axis=-1), atol=1e-5
    )


def test_morton_order_is_deterministic_and_a_permutation():
    points = normalize_unit_sphere(_raw(num=NUM_POINTS))
    first = morton_order(points)
    second = morton_order(points)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(NUM_POINTS))


def test_morton_order_groups_spatial_neighbours():
    """Consecutive points after ordering should be closer than random pairs.

    This is the property the patching relies on -- and the reason a GMP-off run
    with Morton ordering is not free of geometric structure.
    """
    points = normalize_unit_sphere(_raw(num=NUM_POINTS, seed=3))
    ordered = points[morton_order(points)]
    consecutive = np.linalg.norm(np.diff(ordered, axis=0), axis=-1).mean()
    rng = np.random.default_rng(4)
    shuffled = points[rng.permutation(NUM_POINTS)]
    random_gap = np.linalg.norm(np.diff(shuffled, axis=0), axis=-1).mean()
    assert consecutive < random_gap / 2


def test_morton_ties_fall_back_to_incoming_order():
    """Stable sort: identical coordinates keep their relative order."""
    points = np.zeros((4, 3), dtype=np.float32)
    points[:, 0] = [0.0, 0.0, 0.5, 0.5]  # two pairs sharing a cell
    order = morton_order(points)
    assert order[0] < order[1] or list(order[:2]) == [0, 1]
    np.testing.assert_array_equal(np.sort(order), np.arange(4))


def test_fixed_random_order_is_stable_across_calls():
    np.testing.assert_array_equal(fixed_random_order(), fixed_random_order())
    assert not np.array_equal(fixed_random_order(), np.arange(NUM_POINTS))


@pytest.mark.parametrize("ordering", ["morton", "random"])
def test_apply_order_is_a_permutation_of_the_same_points(ordering):
    points = normalize_unit_sphere(_raw(num=NUM_POINTS))
    ordered = apply_order(points, ordering)
    assert ordered.shape == points.shape
    # Same multiset of points, just rearranged.
    np.testing.assert_allclose(
        np.sort(ordered, axis=0), np.sort(points, axis=0), atol=1e-6
    )


def test_apply_order_rejects_unknown_ordering():
    with pytest.raises(ValueError):
        apply_order(normalize_unit_sphere(_raw(num=8)), "kt")


def test_train_sample_shape_and_variation():
    rng = np.random.default_rng(5)
    raw = _raw()
    first = prepare_train_sample(raw, rng, "morton")
    second = prepare_train_sample(raw, rng, "morton")
    assert first.shape == (NUM_POINTS, 3)
    assert first.dtype == np.float32
    assert not np.allclose(first, second), "augmentation should vary per epoch"


def test_eval_sample_is_deterministic():
    """Same object + same indices => identical tensor, every run."""
    raw = _raw()
    idx = eval_subsample_indices(1, raw.shape[0])[0]
    first = prepare_eval_sample(raw, idx, "morton")
    second = prepare_eval_sample(raw, idx, "morton")
    np.testing.assert_array_equal(first, second)
    assert first.shape == (NUM_POINTS, 3)


def test_eval_subsample_indices_are_reproducible_and_unique():
    first = eval_subsample_indices(20, 2048)
    second = eval_subsample_indices(20, 2048)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, NUM_POINTS)
    for row in first:
        assert len(np.unique(row)) == NUM_POINTS, "sampling must be without replacement"


def test_stratified_split_is_disjoint_and_covers_every_class():
    rng = np.random.default_rng(6)
    labels = rng.integers(0, NUM_CLASSES, size=11416)
    train_idx, val_idx = stratified_train_val_split(labels)

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(labels)
    assert set(np.unique(labels[val_idx])) == set(range(NUM_CLASSES))
    assert 0.08 < len(val_idx) / len(labels) < 0.12
    # Deterministic.
    again_train, again_val = stratified_train_val_split(labels)
    np.testing.assert_array_equal(train_idx, again_train)
    np.testing.assert_array_equal(val_idx, again_val)
