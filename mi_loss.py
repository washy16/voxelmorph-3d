import tensorflow as tf

def mutual_information_loss(x, y, bins=32):

    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)
    y = (y - tf.reduce_min(y)) / (tf.reduce_max(y) - tf.reduce_min(y) + 1e-8)

    x_bin = tf.cast(x * (bins - 1), tf.int32)
    y_bin = tf.cast(y * (bins - 1), tf.int32)

    joint_hist = tf.math.bincount(
        x_bin * bins + y_bin,
        minlength=bins*bins,
        maxlength=bins*bins
    )

    joint_hist = tf.reshape(joint_hist, (bins, bins))
    joint_prob = joint_hist / tf.reduce_sum(joint_hist)

    px = tf.reduce_sum(joint_prob, axis=1)
    py = tf.reduce_sum(joint_prob, axis=0)

    px_py = tf.tensordot(px, py, axes=0)

    nz = joint_prob > 0

    mi = tf.reduce_sum(
        tf.where(
            nz,
            joint_prob * tf.math.log(joint_prob / (px_py + 1e-8) + 1e-8),
            0.0
        )
    )

    return -mi