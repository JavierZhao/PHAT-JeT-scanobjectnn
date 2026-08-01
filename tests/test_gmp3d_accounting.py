import pytest

from models.gmp3d_accounting import peak_tensor_bytes


def test_sparse_peak_is_independent_of_grid_side():
    args = dict(batch=32, points=1024, channels=128)
    for variant in ("sparse", "sparse_mean", "sparse_trilinear"):
        assert peak_tensor_bytes(variant, grid_side=36, **args) == peak_tensor_bytes(
            variant, grid_side=71, **args
        )


def test_dense_peak_scales_cubically():
    args = dict(variant="dense", batch=32, points=1024, channels=128)
    small = peak_tensor_bytes(grid_side=36, **args)
    fine = peak_tensor_bytes(grid_side=71, **args)
    assert fine / small == pytest.approx((71 / 36) ** 3)
