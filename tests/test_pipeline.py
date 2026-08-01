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
    load_h5,
    morton_order,
    normalize_unit_sphere,
    prepare_eval_sample,
    prepare_train_sample,
    random_rotation_y,
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


def test_normalize_batched_matches_per_cloud_calls():
    raw = np.stack([_raw(seed=seed) for seed in range(3)])
    expected = np.stack([normalize_unit_sphere(cloud) for cloud in raw])
    np.testing.assert_allclose(normalize_unit_sphere(raw), expected, atol=1e-6)


def test_augmentation_stays_within_the_delta_memory_assumption():
    """Scale <=1.1 on a unit sphere keeps coords in [-1.1, 1.1]."""
    rng = np.random.default_rng(1)
    points = normalize_unit_sphere(_raw())
    for _ in range(50):
        augmented = random_rotation_y(random_scale(points, rng), rng)
        assert np.abs(augmented).max() <= 1.1 + 1e-5


def test_rotation_preserves_shape_and_y_up_axis():
    rng = np.random.default_rng(2)
    points = normalize_unit_sphere(_raw())
    rotated = random_rotation_y(points, rng)
    assert rotated.shape == points.shape
    np.testing.assert_allclose(rotated[:, 1], points[:, 1], atol=1e-6)
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


def test_height_append_is_raw_y_scaled_by_the_same_augmentation_factor():
    """Height is the raw (un-normalized) y, so it carries absolute object size
    -- PointNeXt appends it precisely to make the network "aware of the actual
    size", which unit-sphere normalization destroys.

    But it must be scaled by the same factor as the geometry: scale
    augmentation simulates a differently sized object, so its height changes
    too. Leaving height unscaled would hand the model a height contradicting
    the points beside it, and the ratio between them would leak the
    label-irrelevant augmentation factor.
    """
    raw = _raw()
    expected_rng = np.random.default_rng(23)
    idx = expected_rng.choice(raw.shape[0], size=NUM_POINTS, replace=False)
    scale = expected_rng.uniform(0.9, 1.1)  # same draw order as the pipeline

    actual = prepare_train_sample(
        raw, np.random.default_rng(23), "random",
        fixed_perm=np.arange(NUM_POINTS), height_append=True,
    )
    assert actual.shape == (NUM_POINTS, 4)
    np.testing.assert_allclose(actual[:, 3], raw[idx, 1] * scale, rtol=1e-6)


def test_appended_height_is_invariant_to_y_rotation():
    raw = _raw(num=NUM_POINTS)
    features = np.concatenate(
        [normalize_unit_sphere(raw), raw[:, 1:2]], axis=-1
    )
    rotated = np.concatenate(
        [random_rotation_y(features[:, :3], np.random.default_rng(24)),
         features[:, 3:4]],
        axis=-1,
    )
    np.testing.assert_array_equal(rotated[:, 3], features[:, 3])


def test_eval_height_append_is_raw_y_with_matching_order():
    raw = _raw()
    idx = eval_subsample_indices(1, raw.shape[0])[0]
    actual = prepare_eval_sample(
        raw, idx, "random", fixed_perm=np.arange(NUM_POINTS), height_append=True
    )
    np.testing.assert_array_equal(actual[:, 3], raw[idx, 1])


def test_train_sample_preserves_y_coordinates_except_isotropic_scale():
    """Regression: augmentation rotates around y, never z."""
    raw = _raw()
    rng_expected = np.random.default_rng(17)
    idx = rng_expected.choice(raw.shape[0], size=NUM_POINTS, replace=False)
    normalized = normalize_unit_sphere(raw[idx].astype(np.float32))
    scale = rng_expected.uniform(0.9, 1.1)
    rng_actual = np.random.default_rng(17)
    prepared = prepare_train_sample(raw, rng_actual, "random", fixed_perm=np.arange(NUM_POINTS))
    np.testing.assert_allclose(prepared[:, 1], normalized[:, 1] * scale, atol=1e-6)


def test_epoch_object_shuffle_keeps_labels_aligned():
    labels = np.arange(NUM_CLASSES)
    clouds = np.stack([_raw(num=NUM_POINTS, seed=i) for i in labels])
    rng = np.random.default_rng(18)
    order = rng.permutation(len(labels))
    shuffled_clouds = clouds[order]
    shuffled_labels = labels[order]
    for cloud, label in zip(shuffled_clouds, shuffled_labels):
        np.testing.assert_array_equal(cloud, clouds[label])


@pytest.mark.parametrize("label_shape", [(7,), (7, 1)])
def test_load_h5_canonicalizes_labels_to_vector(tmp_path, label_shape):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    expected = np.arange(7, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle["data"] = np.zeros((7, 8, 3), dtype=np.float32)
        handle["label"] = expected.reshape(label_shape)
    _, labels = load_h5(path)
    assert labels.shape == (7,)
    np.testing.assert_array_equal(labels, expected)


def test_sanity_model_has_pointnet_capacity_and_batch_norm():
    from scripts.sanity_baseline import build_pointnet

    model = build_pointnet(num_points=32)
    batch_norms = [layer for layer in model.layers
                   if layer.__class__.__name__ == "BatchNormalization"]
    assert len(batch_norms) == 7
    assert model.count_params() > 750_000


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
