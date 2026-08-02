"""PHAT-JeT for ScanObjectNN point clouds (plan section 3.3).

The PHAT block structure is unchanged from the jet model -- GMP -> local patch
attention -> patch-token global attention -> broadcast -> FFN. Only the
positional prior (2D -> 3D voxel grid) and the coordinate width (2 -> 3) differ,
plus a classification head sized for 15 classes.

GMP is the only source of positional information: there is deliberately no
separate positional encoding.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from models.gmp3d import GeometricMessagePassing3D
from models.hierarchy import GeometricPooling3D, stage_grid_size
from models.PHAT_JeT import PatchedAttention, PatchMessageBroadcast

# Config ladder. Kept exactly as specified in the handoff; the measured
# parameter counts overshoot the nominal targets and are reported as measured
# rather than retuned (plan section 1.2).
CONFIGS = {
    "XS": dict(d_model=64, blocks=2, heads=4, patch_size=32),
    "S": dict(d_model=128, blocks=2, heads=4, patch_size=32),
    "M": dict(d_model=128, blocks=4, heads=8, patch_size=32),
    "L": dict(d_model=192, blocks=4, heads=8, patch_size=64),
}


class PHATBlock3D(layers.Layer):
    """PHAT block with a 3D voxel GMP prior. `use_gmp=False` is the ablation."""

    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        patch_size,
        grid_size=0.25,
        gmp_kernel=3,
        gmp_variant="dense",
        use_gmp=True,
        dropout=0.0,
        ffn_activation="gelu",
        patch_tokenizer_mode="mean",
        message_proj=True,
        message_gated=False,
        patch_shift=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert ffn_activation in ("relu", "gelu")

        self.use_gmp = use_gmp
        self.patch_shift = int(patch_shift)
        if self.patch_shift < 0 or self.patch_shift >= patch_size:
            raise ValueError("patch_shift must be in [0, patch_size)")
        if use_gmp:
            self.gmp = GeometricMessagePassing3D(
                d_model,
                kernel_size=gmp_kernel,
                grid_size=grid_size,
                variant=gmp_variant,
            )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = PatchedAttention(
            d_model, num_heads, patch_size, dropout=dropout, coord_dim=3
        )
        self.drop1 = layers.Dropout(dropout)

        self.patch_msg = PatchMessageBroadcast(
            d_model=d_model,
            num_heads=num_heads,
            patch_size=patch_size,
            tokenizer_mode=patch_tokenizer_mode,
            dropout=dropout,
            message_proj=message_proj,
            gated=message_gated,
            coord_dim=3,
            name="patch_message",
        )
        self.drop_msg = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(d_ff, activation=ffn_activation),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, coords = inputs

        if self.use_gmp:
            x = self.gmp(x, coords)

        # Swin-style cyclic shifts make adjacent blocks use overlapping point
        # neighbourhoods.  Undo the shift before returning so the public point
        # order and the residual stream are unchanged.
        if self.patch_shift:
            x = tf.roll(x, shift=-self.patch_shift, axis=1)
            coords = tf.roll(coords, shift=-self.patch_shift, axis=1)

        y = self.attn(self.norm1(x), coords, training=training)
        x = x + self.drop1(y, training=training)

        m = self.patch_msg(self.norm1(x), coords, training=training)
        x = x + self.drop_msg(m, training=training)

        y = self.ffn(self.norm2(x), training=training)
        x = x + self.drop2(y, training=training)

        if self.patch_shift:
            x = tf.roll(x, shift=self.patch_shift, axis=1)
            coords = tf.roll(coords, shift=self.patch_shift, axis=1)

        return [x, coords]


class CoarsePHATPath(layers.Layer):
    """One pooled PHAT stage whose messages are broadcast to full resolution.

    Morton-contiguous groups are mean-pooled, processed by an otherwise stock
    PHAT block, then repeated back onto the corresponding fine points.  This
    restores the coarse stage present in the jet architecture without replacing
    the full-resolution PHAT stream or changing its point order.
    """

    def __init__(self, d_model, d_ff, num_heads, pool_size, patch_size,
                 grid_size, gmp_variant="dense", use_gmp=True, dropout=0.0,
                 ffn_activation="gelu", **kwargs):
        super().__init__(**kwargs)
        if pool_size <= 1:
            raise ValueError("hierarchy_pool_size must be greater than 1")
        self.pool_size = int(pool_size)
        self.coarse_block = PHATBlock3D(
            d_model=d_model, d_ff=d_ff, num_heads=num_heads,
            patch_size=patch_size, grid_size=grid_size,
            gmp_variant=gmp_variant, use_gmp=use_gmp, dropout=dropout,
            ffn_activation=ffn_activation, name="coarse_phat_block",
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        x, coords = inputs
        n = x.shape[1]
        if n is None or n % self.pool_size:
            raise ValueError("number of points must be statically divisible by hierarchy_pool_size")
        groups = n // self.pool_size
        coarse_x = tf.reduce_mean(
            tf.reshape(x, [-1, groups, self.pool_size, x.shape[-1]]), axis=2
        )
        coarse_coords = tf.reduce_mean(
            tf.reshape(coords, [-1, groups, self.pool_size, 3]), axis=2
        )
        coarse_x, _ = self.coarse_block([coarse_x, coarse_coords], training=training)
        message = tf.repeat(coarse_x, repeats=self.pool_size, axis=1)
        return [x + self.norm(message), coords]


def build_phat_sonn_classifier(
    config="S",
    num_points=1024,
    num_classes=15,
    grid_size=0.25,
    use_gmp=True,
    gmp_variant="dense",
    patch_size=None,
    dropout=0.0,
    ffn_activation="gelu",
    shifted_patches=False,
    hierarchy_pool_size=None,
    hierarchy_after_block=None,
    downsample_stride=None,
    delta_growth="density",
    gmp_kernel=3,
):
    """Build the point-cloud classifier.

    Args:
        config: one of CONFIGS ("XS", "S", "M", "L").
        grid_size: GMP voxel edge length (delta). Ignored when use_gmp is False.
        use_gmp: False removes the explicit GMP positional prior. Note that
            Morton ordering, if used, still groups spatially near points into
            patches -- this ablation removes the GMP module, not all geometry.
        gmp_variant: ``dense`` (legacy default), ``sparse``, ``sparse_mean``,
            or ``sparse_trilinear``.
        patch_size: override the config's patch size (Phase D).
        shifted_patches: shift odd blocks by half a patch, then restore order.
        hierarchy_pool_size: opt-in pooled PHAT path group size. ``None`` keeps
            the historical architecture exactly unchanged.
        hierarchy_after_block: zero-based block after which to insert the
            coarse path; defaults to the midpoint.
        downsample_stride: opt-in TRUE hierarchy -- actually reduce the point
            count by this factor at the midpoint, as the jet model's
            GeometricPooling did between stages. Distinct from
            hierarchy_pool_size, which keeps full resolution and only adds a
            parallel coarse path. ``None`` leaves the architecture unchanged.
        delta_growth: how the GMP voxel grows after downsampling --
            "density" (edge x stride**(1/3), keeps points-per-voxel constant),
            "double" (PointNeXt's literal radius doubling), or "none"
            (ablation isolating hierarchy from receptive-field scaling).
        gmp_kernel: GMP convolution kernel size. This is the GMP *spatial
            extent* and is independent of delta, which sets resolution. Phase D
            varied the local-attention window and found no effect, but that is
            a different receptive field from this one -- delta changes extent
            and resolution together, so kernel size is the only knob that
            isolates extent.

    Returns:
        keras Model mapping [B, num_points, 3] -> [B, num_classes] logits.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {list(CONFIGS)}")
    cfg = CONFIGS[config]
    d_model = cfg["d_model"]
    patch = cfg["patch_size"] if patch_size is None else patch_size
    hierarchy_at = (cfg["blocks"] // 2 - 1 if hierarchy_after_block is None
                    else hierarchy_after_block)
    if hierarchy_pool_size is not None and not 0 <= hierarchy_at < cfg["blocks"]:
        raise ValueError("hierarchy_after_block is outside the model")

    points = layers.Input((num_points, 3), name="points")
    coords = points  # GMP and patch coords both read raw xyz

    # Input embedding on raw xyz. No positional encoding is added: providing
    # that prior is GMP's job, and testing it is the point of the study.
    x = layers.Dense(d_model, name="input_embedding")(points)

    downsample_at = cfg["blocks"] // 2 - 1
    stage_delta = grid_size
    live_points = num_points
    live_patch = patch

    for i in range(cfg["blocks"]):
        x, coords = PHATBlock3D(
            d_model=d_model,
            d_ff=d_model * 4,
            num_heads=cfg["heads"],
            patch_size=live_patch,
            grid_size=stage_delta,
            gmp_kernel=gmp_kernel,
            use_gmp=use_gmp,
            gmp_variant=gmp_variant,
            dropout=dropout,
            ffn_activation=ffn_activation,
            patch_shift=(patch // 2 if shifted_patches and i % 2 else 0),
            name=f"phat_block3d_{i}",
        )([x, coords])
        if hierarchy_pool_size is not None and i == hierarchy_at:
            coarse_points = num_points // hierarchy_pool_size
            coarse_patch = min(patch, coarse_points)
            if num_points % hierarchy_pool_size or coarse_points % coarse_patch:
                raise ValueError("hierarchy pooling must divide points and yield complete coarse patches")
            x, coords = CoarsePHATPath(
                d_model=d_model, d_ff=d_model * 4, num_heads=cfg["heads"],
                pool_size=hierarchy_pool_size, patch_size=coarse_patch,
                grid_size=grid_size, gmp_variant=gmp_variant, use_gmp=use_gmp,
                dropout=dropout, ffn_activation=ffn_activation,
                name="coarse_phat_path",
            )([x, coords])

        if downsample_stride is not None and i == downsample_at:
            if live_points % downsample_stride:
                raise ValueError(
                    f"point count {live_points} is not divisible by "
                    f"downsample_stride {downsample_stride}"
                )
            x, coords = GeometricPooling3D(
                out_dim=d_model, stride=downsample_stride,
                name="geometric_pool3d",
            )([x, coords])
            live_points //= downsample_stride
            live_patch = min(live_patch, live_points)
            # Fewer points means lower density; grow the voxel so the 3x3x3
            # neighbourhood keeps spanning a comparable number of points.
            stage_delta = stage_grid_size(
                grid_size, stage=1, stride=downsample_stride, growth=delta_growth
            )

    x = layers.GlobalAveragePooling1D(name="global_mean_pool")(x)
    x = layers.Dense(d_model // 2, activation="relu", name="head_hidden")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    # Logits: the loss applies softmax (from_logits=True) so label smoothing is
    # numerically well behaved.
    logits = layers.Dense(num_classes, name="logits")(x)

    return Model(inputs=points, outputs=logits, name=f"phat_sonn_{config}")


def count_flops(model, num_points=1024):
    """Forward FLOPs for one sample, via the TF profiler."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    @tf.function
    def model_fn(x):
        return model(x)

    concrete = model_fn.get_concrete_function(
        tf.TensorSpec((1, num_points, 3), tf.float32)
    )
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        profile = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts
        )
        return profile.total_float_ops
