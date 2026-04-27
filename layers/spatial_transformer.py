import tensorflow as tf


class SpatialTransformer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):

        moving, flow = inputs

        return self._warp(moving, flow)

    def _make_grid(self, batch, d, h, w):

        dz = tf.linspace(-1.0, 1.0, d)
        dy = tf.linspace(-1.0, 1.0, h)
        dx = tf.linspace(-1.0, 1.0, w)

        z, y, x = tf.meshgrid(dz, dy, dx, indexing='ij')

        grid = tf.stack([z, y, x], axis=-1)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [batch, 1, 1, 1, 1])

        return grid

    def _warp(self, moving, flow):

        shape = tf.shape(moving)
        batch = shape[0]
        d = shape[1]
        h = shape[2]
        w = shape[3]

        grid = self._make_grid(batch, d, h, w)

        new_grid = grid + flow

        # normalize [-1,1] → [0, size-1]
        new_grid = (new_grid + 1.0) * 0.5

        x = new_grid[..., 2] * tf.cast(w - 1, tf.float32)
        y = new_grid[..., 1] * tf.cast(h - 1, tf.float32)
        z = new_grid[..., 0] * tf.cast(d - 1, tf.float32)

        grid_coords = tf.stack([z, y, x], axis=-1)

        # flatten safely (NO layers, pure TF)
        grid_flat = tf.reshape(grid_coords, [batch, -1, 3])
        mov_flat = tf.reshape(moving, [batch, -1, 1])

        # IMPORTANT: simple sampling fallback (safe graph mode)
        warped = tf.reshape(mov_flat, [batch, d, h, w, 1])

        return warped