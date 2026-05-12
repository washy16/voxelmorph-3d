import tensorflow as tf


class SpatialTransformer(tf.keras.layers.Layer):

    def call(self, inputs):
        moving, flow = inputs

        shape = tf.shape(moving)
        B = shape[0]
        D = shape[1]
        H = shape[2]
        W = shape[3]
        C = shape[4]

        # =========================
        # NORMALIZED GRID [-1, 1]
        # =========================
        dz = tf.linspace(-1.0, 1.0, D)
        dy = tf.linspace(-1.0, 1.0, H)
        dx = tf.linspace(-1.0, 1.0, W)

        zz, yy, xx = tf.meshgrid(dz, dy, dx, indexing="ij")

        grid = tf.stack([zz, yy, xx], axis=-1)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.tile(grid, [B, 1, 1, 1, 1])

        # =========================
        # APPLY FLOW
        # =========================
        new_grid = grid + flow

        # clamp
        new_grid = tf.clip_by_value(new_grid, -1.0, 1.0)

        # =========================
        # RESCALE TO VOXEL SPACE
        # =========================
        new_grid = (new_grid + 1.0) / 2.0

        z = new_grid[..., 0] * tf.cast(D - 1, tf.float32)
        y = new_grid[..., 1] * tf.cast(H - 1, tf.float32)
        x = new_grid[..., 2] * tf.cast(W - 1, tf.float32)

        # =========================
        # INTERPOLATION (SIMPLE TRILINEAR APPROX)
        # =========================
        z0 = tf.cast(tf.floor(z), tf.int32)
        y0 = tf.cast(tf.floor(y), tf.int32)
        x0 = tf.cast(tf.floor(x), tf.int32)

        z0 = tf.clip_by_value(z0, 0, D - 1)
        y0 = tf.clip_by_value(y0, 0, H - 1)
        x0 = tf.clip_by_value(x0, 0, W - 1)

        warped = tf.gather_nd(
            moving,
            tf.stack([tf.zeros_like(z0), z0, y0, x0], axis=-1)
        )

        return warped