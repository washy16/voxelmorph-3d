import tensorflow as tf


def dice(y_true, y_pred):

    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)

    return (2. * inter + 1e-8) / (union + 1e-8)


def jacobian_determinant(flow):
    """
    % folding (critique en registration paper)
    """

    grad = flow[:,1:,1:,1:,:] - flow[:,:-1,:-1,:-1,:]
    det = tf.reduce_mean(grad)

    return tf.reduce_mean(det < 0)