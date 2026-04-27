import tensorflow as tf


class DiffeomorphicIntegration(tf.keras.layers.Layer):
    """
    Version 3D SAFE (sans tf.image.resize).
    """

    def __init__(self, scale_factor=2, **kwargs):
        super(DiffeomorphicIntegration, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, flow):
        """
        flow: (B, H, W, D, 3)
        """

        # Upsampling 3D correct
        flow_upsampled = tf.keras.layers.UpSampling3D(
            size=(self.scale_factor,
                  self.scale_factor,
                  self.scale_factor)
        )(flow)

        # ⚠️ On évite tf.image.resize COMPLETEMENT
        # On force juste une cohérence de shape si nécessaire

        if flow_upsampled.shape[1:4] != flow.shape[1:4]:
            flow_upsampled = tf.keras.layers.Cropping3D(
                cropping=((0, flow_upsampled.shape[1] - flow.shape[1]),
                          (0, flow_upsampled.shape[2] - flow.shape[2]),
                          (0, flow_upsampled.shape[3] - flow.shape[3]))
            )(flow_upsampled)

        # Approx integration simple (type VoxelMorph baseline)
        return flow + flow_upsampled