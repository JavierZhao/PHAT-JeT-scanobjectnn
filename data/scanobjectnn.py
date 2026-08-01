"""ScanObjectNN PB_T50_RS pipeline (plan section 3.1 / 3.3).

Protocol follows the PointNeXt/PointMLP convention: 1024 points, unit-sphere
normalized coordinates, scale+rotation augmentation at train time, and no
voting at test time.  PointNeXt-style raw height is optionally appended.

All array transforms here are pure numpy so they can be unit-tested without the
dataset, which is license-gated and must be staged on the PVC by hand.
"""

import os

import numpy as np

NUM_CLASSES = 15
NUM_POINTS = 1024
SPLIT_SEED = 42  # train/val split, eval subsample indices, random ordering

TRAIN_FILE = "training_objectdataset_augmentedrot_scale75.h5"
TEST_FILE = "test_objectdataset_augmentedrot_scale75.h5"
EXPECTED_TRAIN = 11416
EXPECTED_TEST = 2882

# Morton ordering grid: fixed bounds, independent of the GMP delta.
MORTON_BITS = 6  # 64 cells per axis
MORTON_LO, MORTON_HI = -1.1, 1.1  # unit sphere inflated by the 1.1 scale aug


# --------------------------------------------------------------------------
# Normalization and augmentation
# --------------------------------------------------------------------------
def normalize_unit_sphere(points):
    """Zero mean, unit max-radius. points: [..., N, 3] -> same shape."""
    centered = points - points.mean(axis=-2, keepdims=True)
    radius = np.linalg.norm(centered, axis=-1).max(axis=-1, keepdims=True)[..., None]
    return centered / np.maximum(radius, 1e-8)


def random_scale(points, rng, low=0.9, high=1.1, return_factor=False):
    """Isotropic random scaling, one factor per cloud.

    `return_factor` exposes the drawn factor so callers can apply the same
    scaling to derived features (e.g. appended height). The RNG draw order is
    unchanged, so results are identical to the previous implementation.
    """
    factor = rng.uniform(low, high)
    scaled = points * factor
    return (scaled, factor) if return_factor else scaled


def random_rotation_y(points, rng):
    """Random rotation about ScanObjectNN's y-up axis."""
    theta = rng.uniform(0, 2 * np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    matrix = np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float32
    )
    return points @ matrix.T


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------
def _interleave3(x, y, z, bits=MORTON_BITS):
    """Interleave the low `bits` of three non-negative integer arrays."""
    x = x.astype(np.uint64)
    y = y.astype(np.uint64)
    z = z.astype(np.uint64)
    key = np.zeros_like(x, dtype=np.uint64)
    for i in range(bits):
        key |= ((x >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i)
        key |= ((y >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i + 1)
        key |= ((z >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i + 2)
    return key


def morton_order(points, bits=MORTON_BITS):
    """Indices sorting points along a 3D Morton (Z-order) curve.

    Fixed quantization bounds so the mapping does not depend on the cloud, and
    a stable sort so ties (points sharing a cell) fall back to the incoming
    order, which is itself seeded and therefore reproducible.

    points: [N, 3] -> int64 [N]
    """
    span = MORTON_HI - MORTON_LO
    cells = 1 << bits
    quantized = np.floor((points - MORTON_LO) / span * cells).astype(np.int64)
    quantized = np.clip(quantized, 0, cells - 1)
    key = _interleave3(quantized[:, 0], quantized[:, 1], quantized[:, 2], bits)
    return np.argsort(key, kind="stable")


def fixed_random_order(num_points=NUM_POINTS, seed=SPLIT_SEED):
    """One permutation, drawn once, applied to every cloud, train and test."""
    return np.random.default_rng(seed).permutation(num_points)


def apply_order(points, ordering, fixed_perm=None):
    """Reorder a single cloud. points: [N, 3] -> [N, 3]."""
    if ordering == "morton":
        return points[morton_order(points)]
    if ordering == "random":
        perm = fixed_random_order(points.shape[0]) if fixed_perm is None else fixed_perm
        return points[perm]
    raise ValueError(f"unknown ordering: {ordering!r}")


# --------------------------------------------------------------------------
# Sample-level composition
# --------------------------------------------------------------------------
def prepare_train_sample(raw_points, rng, ordering, fixed_perm=None,
                         num_points=NUM_POINTS, height_append=False):
    """Resample -> normalize -> augment -> order. raw_points: [2048, 3]."""
    idx = rng.choice(raw_points.shape[0], size=num_points, replace=False)
    selected = raw_points[idx].astype(np.float32)
    # PointNeXt height is the raw gravity-axis measurement.  Capture it before
    # centering/unit-sphere normalization removes absolute height and scale.
    height = selected[:, 1:2]
    points, scale = random_scale(normalize_unit_sphere(selected), rng,
                                 return_factor=True)
    points = random_rotation_y(points, rng)
    if height_append:
        # The same scale factor must apply to the height. Scale augmentation
        # simulates a differently sized object, so its height changes too;
        # leaving height unscaled would feed the model a height that
        # contradicts the geometry beside it.
        points = np.concatenate([points, height * scale], axis=-1)
    # Ordering must come after augmentation: rotation changes Morton codes.
    return apply_order(points, ordering, fixed_perm).astype(np.float32)


def prepare_eval_sample(raw_points, subsample_idx, ordering, fixed_perm=None,
                        height_append=False):
    """Fixed subsample -> normalize -> order. No augmentation, no voting."""
    selected = raw_points[subsample_idx].astype(np.float32)
    height = selected[:, 1:2]
    points = normalize_unit_sphere(selected)
    if height_append:
        points = np.concatenate([points, height], axis=-1)
    return apply_order(points, ordering, fixed_perm).astype(np.float32)


def eval_subsample_indices(num_objects, raw_num_points, num_points=NUM_POINTS,
                           seed=SPLIT_SEED):
    """Deterministic per-object point subset for val/test.

    Drawn once, in file order, and reused byevery model, phase and ablation so
    all reported numbers are computed on identical points.
    """
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.choice(raw_num_points, size=num_points, replace=False)
         for _ in range(num_objects)]
    )


def stratified_train_val_split(labels, val_fraction=0.1, seed=SPLIT_SEED):
    """Class-balanced split. Val is used only for checkpoint selection."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        rng.shuffle(members)
        cut = max(1, int(round(len(members) * val_fraction)))
        val_idx.append(members[:cut])
        train_idx.append(members[cut:])
    return (
        np.sort(np.concatenate(train_idx)),
        np.sort(np.concatenate(val_idx)),
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_h5(path):
    """Read one ScanObjectNN h5 file -> (points [M, 2048, 3], labels [M])."""
    import h5py

    with h5py.File(path, "r") as handle:
        points = np.array(handle["data"]).astype(np.float32)
        # ScanObjectNN mirrors exist with both [M] and [M, 1] label datasets.
        # Canonicalize at the boundary: leaving [M, 1] intact makes a later
        # comparison against predictions [M] silently broadcast to [M, M].
        labels = np.array(handle["label"]).astype(np.int64).reshape(-1)
    return points, labels


def load_split(data_dir, split):
    """Load and validate a split. Fails loudly on anything unexpected."""
    filename = TRAIN_FILE if split == "train" else TEST_FILE
    expected = EXPECTED_TRAIN if split == "train" else EXPECTED_TEST
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. The PB_T50_RS files are license-gated: request "
            "them via the ScanObjectNN terms-of-use form and stage them on the "
            "PVC (see k8s_staging_pod.yaml)."
        )

    points, labels = load_h5(path)
    if points.shape[0] != expected:
        raise ValueError(
            f"{filename}: expected {expected} objects, found {points.shape[0]}. "
            "Wrong variant? PB_T50_RS is *_augmentedrot_scale75.h5."
        )
    if points.shape[2] != 3:
        raise ValueError(f"{filename}: expected xyz coordinates, got {points.shape}")
    classes = np.unique(labels)
    if classes.min() < 0 or classes.max() >= NUM_CLASSES:
        raise ValueError(f"{filename}: labels outside [0, {NUM_CLASSES}) -> {classes}")
    if split == "test" and len(classes) != NUM_CLASSES:
        raise ValueError(
            f"test split covers {len(classes)} classes, expected {NUM_CLASSES}; "
            "mAcc would be undefined."
        )
    return points, labels
