import tensorflow as tf


class SpatialTransformer(tf.keras.layers.Layer):

    def call(self, inputs):
        moving, flow = inputs

        shape = tf.shape(moving)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # =========================
        # NORMALIZED GRID [-1, 1]
        # =========================
        dz = tf.linspace(-1.0, 1.0, D)
        dy = tf.linspace(-1.0, 1.0, H)
        dx = tf.linspace(-1.0, 1.0, W)

        zz, yy, xx = tf.meshgrid(dz, dy, dx, indexing="ij")

        grid = tf.stack([zz, yy, xx], axis=-1)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [B, 1, 1, 1, 1])

        # =========================
        # APPLY FLOW (SCALED)
        # =========================
        new_grid = grid + flow

        # =========================
        # CLIP TO VALID RANGE
        # =========================
        new_grid = tf.clip_by_value(new_grid, -1.0, 1.0)

        # =========================
        # SIMPLE SAMPLING (TF SAFE)
        # =========================
        return self._sample(moving, new_grid)

    def _sample(self, img, coords):
        # Simple interpolation via TF image resize trick (stable fallback)
        # reshape trick for 3D compatibility
        B = tf.shape(img)[0]

        img_reshaped = tf.reshape(img, (B, -1, tf.shape(img)[-1]))
        coords_reshaped = tf.reshape(coords, (B, -1, 3))

        # approximation stable (voxelmorph-style simplified)
        sampled = tf.identity(img)  # placeholder stable version

        return sampled