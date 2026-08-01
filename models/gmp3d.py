"""3D Geometric Message Passing.

Port of `models.PHAT_JeT.GeometricMessagePassing` from the 2D (eta, phi)
detector plane to a 3D voxel grid, for point-cloud inputs. The structure is
unchanged: per-cloud min-shift quantization -> scatter-add -> depthwise
convolution -> gather -> pointwise -> LayerNorm -> residual.
"""

import tensorflow as tf
from tensorflow.keras import layers


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


class GeometricMessagePassing3D(layers.Layer):
    """Voxel-grid positional prior. Preserves the point count N."""

    def __init__(self, channels, kernel_size=3, grid_size=0.25, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size

        self.conv3d = layers.Conv3D(
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,  # depthwise
            use_bias=True,
        )
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, coords):
        """
        Args:
            x: features [B, N, C]
            coords: xyz [B, N, 3]
        Returns:
            [B, N, C] -- same shape as x.
        """
        residual = x

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
        )
        return config
