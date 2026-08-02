"""True hierarchical downsampling for the 3D PHAT model.

Motivation: the scaling ladder is flat (17x parameters buys +2.2 OA, and M ~= L),
while interventions that change *what information reaches the model* -- GMP grid
resolution and height appending -- gave +8.4 and +4.0. That pattern says the
model is not capacity-limited but context-limited.

PointNeXt names exactly two mechanisms for its gains: "(1) using an appropriate
radius to query the neighborhood, and (2) adopting a hierarchical architecture",
with the radius doubling each time the cloud is downsampled. PointNet++,
PointMLP and DGCNN are all hierarchical too. The 3D PHAT model has neither: every
block runs on all 1024 points at one resolution with a fixed 3x3x3 GMP kernel.

Note this is a *restoration*, not an import: the original jet model had
`GeometricPooling` between stages (strides [2, 2], 150 -> 75 -> 38 particles).
It was dropped when the ladder was flattened to a single stage. So adding it back
keeps the transfer study honest -- we are porting the jet architecture more
faithfully, not grafting on a PointNeXt component.
"""

import tensorflow as tf
from tensorflow.keras import layers


class GeometricPooling3D(layers.Layer):
    """Downsample by pooling Morton-contiguous groups of points.

    The jet model's `GeometricPooling` sorted particles by eta, grouped
    consecutive `stride` of them, max-pooled features and mean-pooled
    coordinates. The 3D analogue needs no sort: points arrive Morton-ordered
    from the data pipeline and the model never reorders them, so consecutive
    groups are already spatially coherent.

    Features are max-pooled (as in the jet model) and coordinates mean-pooled,
    giving each surviving point the centroid of the group it summarizes.
    """

    def __init__(self, out_dim, stride=2, **kwargs):
        super().__init__(**kwargs)
        if stride < 2:
            raise ValueError("stride must be >= 2 to downsample")
        self.out_dim = out_dim
        self.stride = int(stride)
        self.proj = layers.Dense(out_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x, coords = inputs
        num_points = x.shape[1]
        if num_points is None or num_points % self.stride:
            raise ValueError(
                f"point count {num_points} must be statically divisible by "
                f"stride {self.stride}"
            )
        groups = num_points // self.stride
        channels = x.shape[-1]

        x = tf.reduce_max(
            tf.reshape(x, [-1, groups, self.stride, channels]), axis=2
        )
        coords = tf.reduce_mean(
            tf.reshape(coords, [-1, groups, self.stride, 3]), axis=2
        )

        x = self.norm(self.proj(x))
        return [x, coords]

    def compute_output_shape(self, input_shape):
        x_shape, coord_shape = input_shape
        groups = None if x_shape[1] is None else x_shape[1] // self.stride
        return [
            (x_shape[0], groups, self.out_dim),
            (coord_shape[0], groups, coord_shape[2]),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(out_dim=self.out_dim, stride=self.stride)
        return config


def stage_grid_size(base_delta, stage, stride, growth="density"):
    """GMP voxel size for a given stage -- receptive-field scaling.

    Downsampling lowers point density, so holding the voxel size fixed leaves
    later stages with mostly-empty voxels and a 3x3x3 kernel spanning almost no
    points. PointNeXt handles the same problem by growing the query radius as
    the cloud is downsampled.

      density -- grow the voxel so points-per-voxel stays constant. Reducing the
                 count by `stride` means the volume should grow by `stride`, so
                 the edge grows by stride**(1/3). This is the conservative,
                 physically motivated choice.
      double  -- double the edge each stage, matching PointNeXt's "radius
                 doubles when the point cloud is downsampled" literally. More
                 aggressive.
      none    -- keep delta fixed (ablation: isolates hierarchy from
                 receptive-field scaling).
    """
    if growth == "none":
        return base_delta
    if growth == "density":
        return base_delta * (float(stride) ** (1.0 / 3.0)) ** stage
    if growth == "double":
        return base_delta * (2.0 ** stage)
    raise ValueError(f"unknown growth {growth!r}; expected density, double or none")
