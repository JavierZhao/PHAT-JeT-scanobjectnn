"""Dependency-free peak-tensor accounting for GMP experiment planning.

This is intentionally an analytical upper bound rather than a process-RSS
benchmark: TensorFlow's CPU allocator retains memory, which makes RSS and
tracemalloc misleading across variants. The estimate covers the dominant GMP
forward tensors and excludes the common PHAT attention/FFN activations.
"""


def peak_tensor_bytes(variant, batch, points, channels, grid_side, dtype_bytes=4):
    """Conservative peak live tensor bytes for one GMP block's forward pass."""
    point_features = batch * points * channels * dtype_bytes
    if variant == "dense":
        grid = batch * grid_side**3 * channels * dtype_bytes
        # Conv3D needs its input and output concurrently.
        return 2 * grid
    if variant in ("sparse", "sparse_mean"):
        # Coalesced values, convolution accumulator, and one neighbor gather.
        return 3 * point_features
    if variant == "sparse_trilinear":
        # At most eight occupied corners per point, with the same three live
        # feature tensors as sparse hard voxelization.
        return 3 * 8 * point_features
    raise ValueError(f"unknown GMP variant {variant!r}")


def gibibytes(byte_count):
    return byte_count / 1024**3
