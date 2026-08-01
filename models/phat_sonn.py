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
        use_gmp=True,
        dropout=0.0,
        ffn_activation="gelu",
        patch_tokenizer_mode="mean",
        message_proj=True,
        message_gated=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert ffn_activation in ("relu", "gelu")

        self.use_gmp = use_gmp
        if use_gmp:
            self.gmp = GeometricMessagePassing3D(
                d_model, kernel_size=gmp_kernel, grid_size=grid_size
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

        y = self.attn(self.norm1(x), coords, training=training)
        x = x + self.drop1(y, training=training)

        m = self.patch_msg(self.norm1(x), coords, training=training)
        x = x + self.drop_msg(m, training=training)

        y = self.ffn(self.norm2(x), training=training)
        x = x + self.drop2(y, training=training)

        return [x, coords]


def build_phat_sonn_classifier(
    config="S",
    num_points=1024,
    num_classes=15,
    grid_size=0.25,
    use_gmp=True,
    patch_size=None,
    dropout=0.0,
    ffn_activation="gelu",
):
    """Build the point-cloud classifier.

    Args:
        config: one of CONFIGS ("XS", "S", "M", "L").
        grid_size: GMP voxel edge length (delta). Ignored when use_gmp is False.
        use_gmp: False removes the explicit GMP positional prior. Note that
            Morton ordering, if used, still groups spatially near points into
            patches -- this ablation removes the GMP module, not all geometry.
        patch_size: override the config's patch size (Phase D).

    Returns:
        keras Model mapping [B, num_points, 3] -> [B, num_classes] logits.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {list(CONFIGS)}")
    cfg = CONFIGS[config]
    d_model = cfg["d_model"]
    patch = cfg["patch_size"] if patch_size is None else patch_size

    points = layers.Input((num_points, 3), name="points")
    coords = points  # GMP and patch coords both read raw xyz

    # Input embedding on raw xyz. No positional encoding is added: providing
    # that prior is GMP's job, and testing it is the point of the study.
    x = layers.Dense(d_model, name="input_embedding")(points)

    for i in range(cfg["blocks"]):
        x, coords = PHATBlock3D(
            d_model=d_model,
            d_ff=d_model * 4,
            num_heads=cfg["heads"],
            patch_size=patch,
            grid_size=grid_size,
            use_gmp=use_gmp,
            dropout=dropout,
            ffn_activation=ffn_activation,
            name=f"phat_block3d_{i}",
        )([x, coords])

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
