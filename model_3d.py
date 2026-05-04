import tensorflow as tf
from config import IMG_SHAPE
from layers.spatial_transformer import SpatialTransformer


# =========================
# BLOCK CONV (U-NET STYLE)
# =========================
def conv_block(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv3D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


# =========================
# DOWN SAMPLING BLOCK
# =========================
def down_block(x, filters):
    f = conv_block(x, filters)
    p = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(f)
    return f, p


# =========================
# UP SAMPLING BLOCK
# =========================
def up_block(x, skip, filters):
    x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


# =========================
# MODEL VOXELMORPH 3D
# =========================
def build_voxelmorph_3d():

    moving = tf.keras.Input(shape=IMG_SHAPE)
    fixed = tf.keras.Input(shape=IMG_SHAPE)

    # =========================
    # INPUT FUSION
    # =========================
    x = tf.keras.layers.Concatenate()([moving, fixed])

    # =========================
    # ENCODER (U-NET)
    # =========================
    s1, p1 = down_block(x, 16)
    s2, p2 = down_block(p1, 32)
    s3, p3 = down_block(p2, 64)

    # bottleneck
    b = conv_block(p3, 128)

    # =========================
    # DECODER
    # =========================
    u1 = up_block(b, s3, 64)
    u2 = up_block(u1, s2, 32)
    u3 = up_block(u2, s1, 16)

    # =========================
    # FLOW FIELD
    # =========================
    flow = tf.keras.layers.Conv3D(
        3,
        kernel_size=3,
        padding="same",
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5)
    )(u3)

    # 🔥 STABILISATION DU FLOW
    flow = tf.keras.layers.Activation("tanh")(flow)

    # scale faible (important pour stabilité)
    flow = tf.keras.layers.Lambda(lambda x: x * 2.0)(flow)

    # =========================
    # WARPING
    # =========================
    warped = SpatialTransformer()([moving, flow])

    return tf.keras.Model(inputs=[moving, fixed], outputs=[warped, flow])