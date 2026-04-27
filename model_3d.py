import tensorflow as tf
from config import IMG_SHAPE
from layers.spatial_transformer import SpatialTransformer


def conv_block(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_voxelmorph_3d():

    moving = tf.keras.Input(shape=IMG_SHAPE)
    fixed = tf.keras.Input(shape=IMG_SHAPE)

    x = tf.keras.layers.Concatenate()([moving, fixed])

    # Encoder U-Net
    x = conv_block(x, 16)
    x = tf.keras.layers.MaxPool3D()(x)

    x = conv_block(x, 32)
    x = tf.keras.layers.MaxPool3D()(x)

    x = conv_block(x, 64)

    # Flow field
    flow = tf.keras.layers.Conv3D(
        3, 3, padding="same", activation=None,
        kernel_initializer="zeros"
    )(x)

    # Resize flow to image size (PROPRE)
    flow = tf.keras.layers.UpSampling3D(size=(4,4,4))(flow)

    # Spatial transform
    warped = SpatialTransformer()([moving, flow])

    return tf.keras.Model(inputs=[moving, fixed], outputs=[warped, flow])