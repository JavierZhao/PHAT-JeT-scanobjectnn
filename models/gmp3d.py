"""3D Geometric Message Passing.

Port of `models.PHAT_JeT.GeometricMessagePassing` from the 2D (eta, phi)
detector plane to a 3D voxel grid, for point-cloud inputs. The structure is
unchanged: per-cloud min-shift quantization -> scatter-add -> depthwise
convolution -> gather -> pointwise -> LayerNorm -> residual.
"""

import tensorflow as tf
from tensorflow.keras import layers


_SPARSE_VARIANTS = ("sparse", "sparse_mean", "sparse_trilinear")


def quantize(coords, grid_size):
    """Per-cloud min-shifted voxel indices.

    Args:
        coords: [B, N, 3] float coordinates.
        grid_size: voxel edge length (delta).

    Returns:
        [B, N, 3] int32 indices, non-negative by construction (the shift uses
        each cloud's own per-axis minimum, exactly as 2D GMP Eq. (1)).
    """
    mins = tf.reduce_min(coords, axis=1, keepdims=True)  # [B, 1, 3]
    return tf.cast(tf.floor((coords - mins) / grid_size), tf.int32)


def scatter_to_grid(features, idx, grid_dims):
    """Sum point features into their voxels.

    Factored out of the layer so the duplicate-voxel summation semantics can be
    unit-tested without convolution/normalization/residual on top.

    Args:
        features: [B, N, C]
        idx: [B, N, 3] non-negative int32 voxel indices
        grid_dims: int32 tensor/list of 3 grid side lengths [Gx, Gy, Gz]

    Returns:
        [B, Gx, Gy, Gz, C]; points landing in the same voxel are summed.
    """
    shape = tf.shape(features)
    batch, num_points, channels = shape[0], shape[1], shape[2]

    batch_idx = tf.tile(tf.range(batch)[:, None], [1, num_points])  # [B, N]
    full_idx = tf.concat([batch_idx[..., None], idx], axis=-1)  # [B, N, 4]

    out_shape = tf.concat(
        [[batch], tf.cast(tf.stack(grid_dims), tf.int32), [channels]], axis=0
    )
    # scatter_nd accumulates duplicate indices, which is the behaviour we want.
    return tf.scatter_nd(
        tf.reshape(full_idx, [-1, 4]),
        tf.reshape(features, [-1, channels]),
        out_shape,
    )


def _corner_offsets(size):
    """Integer offsets in Conv3D kernel (x, y, z) order."""
    axis = tf.range(size, dtype=tf.int32)
    xx, yy, zz = tf.meshgrid(axis, axis, axis, indexing="ij")
    return tf.stack([xx, yy, zz], axis=-1)


def _linear_keys(batch_idx, idx, grid_dims):
    """Collision-free dynamic row-major keys for batched voxel indices."""
    idx = tf.cast(idx, tf.int64)
    dims = tf.cast(grid_dims, tf.int64)
    batch_idx = tf.cast(batch_idx, tf.int64)
    return ((batch_idx * dims[0] + idx[..., 0]) * dims[1] + idx[..., 1]) * dims[2] + idx[..., 2]


def _sparse_entries(features, idx, weights=None, normalize=False):
    """Coalesce point/corner contributions into sorted occupied voxels."""
    shape = tf.shape(features)
    batch, num_points = shape[0], shape[1]
    batch_idx = tf.tile(tf.range(batch)[:, None], [1, num_points])
    grid_dims = tf.reduce_max(idx, axis=[0, 1]) + 1
    keys = _linear_keys(batch_idx, idx, grid_dims)
    flat_keys = tf.reshape(keys, [-1])
    flat_features = tf.reshape(features, [-1, shape[2]])
    if weights is not None:
        flat_weights = tf.reshape(tf.cast(weights, features.dtype), [-1])
        flat_features = flat_features * flat_weights[:, None]
    else:
        flat_weights = tf.ones_like(flat_keys, dtype=features.dtype)

    unique_keys, segments = tf.unique(flat_keys)
    count = tf.shape(unique_keys)[0]
    values = tf.math.unsorted_segment_sum(flat_features, segments, count)
    masses = tf.math.unsorted_segment_sum(flat_weights, segments, count)
    if normalize:
        values = tf.math.divide_no_nan(values, masses[:, None])

    order = tf.argsort(unique_keys)
    unique_keys = tf.gather(unique_keys, order)
    values = tf.gather(values, order)
    # Decode keys so neighbor lookup does not depend on a fixed coordinate cap.
    dims64 = tf.cast(grid_dims, tf.int64)
    z = unique_keys % dims64[2]
    q = unique_keys // dims64[2]
    y = q % dims64[1]
    q = q // dims64[1]
    x = q % dims64[0]
    b = q // dims64[0]
    coords4 = tf.cast(tf.stack([b, x, y, z], axis=-1), tf.int32)
    return unique_keys, values, coords4, grid_dims


def _lookup_sorted(keys, values, query_keys, valid):
    """Hash-table-like lookup implemented with graph-safe sorted tensors."""
    size = tf.shape(keys)[0]
    query_shape = tf.shape(query_keys)
    flat_queries = tf.reshape(query_keys, [-1])
    positions = tf.searchsorted(keys, flat_queries, side="left")
    safe_positions = tf.minimum(positions, size - 1)
    flat_valid = tf.reshape(valid, [-1])
    found = tf.logical_and(
        flat_valid, tf.equal(tf.gather(keys, safe_positions), flat_queries)
    )
    gathered = tf.gather(values, safe_positions)
    gathered = tf.where(found[..., None], gathered, tf.zeros_like(gathered))
    return tf.reshape(
        gathered, tf.concat([query_shape, [tf.shape(values)[-1]]], axis=0)
    )


class GeometricMessagePassing3D(layers.Layer):
    """Voxel-grid positional prior. Preserves the point count N."""

    def __init__(
        self, channels, kernel_size=3, grid_size=0.25, variant="dense", **kwargs
    ):
        super().__init__(**kwargs)
        if variant not in ("dense",) + _SPARSE_VARIANTS:
            raise ValueError(
                f"unknown GMP variant {variant!r}; expected dense or {_SPARSE_VARIANTS}"
            )
        if kernel_size % 2 != 1:
            raise ValueError("GMP kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.variant = variant

        self.conv3d = layers.Conv3D(
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,  # depthwise
            use_bias=True,
        )
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # Sparse variants use Conv3D as an exactly compatible weight container.
        # Building it here makes `kernel` available without allocating a grid.
        if not self.conv3d.built:
            self.conv3d.build((None, None, None, None, self.channels))
        super().build(input_shape)

    def _sparse_convolution(self, keys, values, coords4, grid_dims):
        """Depthwise same convolution evaluated only at occupied voxels."""
        # Grouped Conv3D kernel layout is [K, K, K, 1, C].
        kernel = self.conv3d.kernel[..., 0, :]
        out = tf.zeros_like(values)
        radius = self.kernel_size // 2
        # Unrolling the 27 offsets keeps the largest temporary [M, C], instead
        # of materializing [M, 27, C]. M is the number of occupied voxels.
        for ix in range(self.kernel_size):
            for iy in range(self.kernel_size):
                for iz in range(self.kernel_size):
                    offset = tf.constant([ix - radius, iy - radius, iz - radius])
                    query_xyz = coords4[:, 1:] + offset
                    valid = tf.reduce_all(
                        tf.logical_and(query_xyz >= 0, query_xyz < grid_dims), axis=-1
                    )
                    query_keys = _linear_keys(
                        coords4[:, 0], tf.maximum(query_xyz, 0), grid_dims
                    )
                    neighbors = _lookup_sorted(keys, values, query_keys, valid)
                    out = out + neighbors * kernel[ix, iy, iz][None, :]
        if self.conv3d.use_bias:
            out = out + self.conv3d.bias
        return out

    def _call_sparse(self, x, coords):
        shifted = (coords - tf.reduce_min(coords, axis=1, keepdims=True)) / self.grid_size
        base = tf.cast(tf.floor(shifted), tf.int32)

        if self.variant == "sparse_trilinear":
            offsets = tf.reshape(_corner_offsets(2), [1, 1, 8, 3])
            corner_idx = base[:, :, None, :] + offsets
            fraction = shifted - tf.floor(shifted)
            factors = tf.where(
                tf.equal(offsets, 1), fraction[:, :, None, :], 1.0 - fraction[:, :, None, :]
            )
            weights = tf.reduce_prod(factors, axis=-1)
            tiled_x = tf.broadcast_to(x[:, :, None, :], [tf.shape(x)[0], tf.shape(x)[1], 8, tf.shape(x)[2]])
            flat_idx = tf.reshape(corner_idx, [tf.shape(x)[0], -1, 3])
            flat_x = tf.reshape(tiled_x, [tf.shape(x)[0], -1, tf.shape(x)[2]])
            flat_weights = tf.reshape(weights, [tf.shape(x)[0], -1])
            keys, values, coords4, grid_dims = _sparse_entries(
                flat_x, flat_idx, flat_weights, normalize=False
            )
            voxel_out = self._sparse_convolution(keys, values, coords4, grid_dims)
            point_keys = _linear_keys(
                tf.tile(tf.range(tf.shape(x)[0])[:, None, None], [1, tf.shape(x)[1], 8]),
                corner_idx,
                grid_dims,
            )
            gathered = _lookup_sorted(
                keys, voxel_out, point_keys, tf.ones_like(point_keys, dtype=tf.bool)
            )
            out = tf.reduce_sum(gathered * weights[..., None], axis=2)
        else:
            keys, values, coords4, grid_dims = _sparse_entries(
                x, base, normalize=self.variant == "sparse_mean"
            )
            voxel_out = self._sparse_convolution(keys, values, coords4, grid_dims)
            batch_idx = tf.tile(tf.range(tf.shape(x)[0])[:, None], [1, tf.shape(x)[1]])
            point_keys = _linear_keys(batch_idx, base, grid_dims)
            out = _lookup_sorted(
                keys, voxel_out, point_keys, tf.ones_like(point_keys, dtype=tf.bool)
            )

        out = tf.ensure_shape(out, [None, None, self.channels])
        out = self.pointwise(out)
        out = self.norm(out)
        return x + out

    def call(self, x, coords):
        """
        Args:
            x: features [B, N, C]
            coords: xyz [B, N, 3]
        Returns:
            [B, N, C] -- same shape as x.
        """
        residual = x

        if self.variant != "dense":
            return self._call_sparse(x, coords)

        idx = quantize(coords, self.grid_size)
        # Grid dims taken over the whole batch (may over-allocate for clouds
        # with a smaller extent); mirrors the 2D implementation.
        grid_dims = [tf.reduce_max(idx[..., a]) + 1 for a in range(3)]

        grid = scatter_to_grid(x, idx, grid_dims)
        grid = self.conv3d(grid)

        shape = tf.shape(x)
        batch_idx = tf.tile(tf.range(shape[0])[:, None], [1, shape[1]])
        out = tf.gather_nd(grid, tf.concat([batch_idx[..., None], idx], axis=-1))
        out = tf.ensure_shape(out, [None, None, self.channels])

        out = self.pointwise(out)
        out = self.norm(out)
        return residual + out

    def get_config(self):
        config = super().get_config()
        config.update(
            channels=self.channels,
            kernel_size=self.kernel_size,
            grid_size=self.grid_size,
            variant=self.variant,
        )
        return config
