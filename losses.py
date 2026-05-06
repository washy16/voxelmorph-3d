import tensorflow as tf
from config import LAMBDA_REG


# =========================
# NCC LOSS (STABLE BASELINE)
# =========================
def ncc_loss(y_true, y_pred, eps=1e-5):

    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)

    true_centered = y_true - mean_true
    pred_centered = y_pred - mean_pred

    numerator = tf.reduce_mean(true_centered * pred_centered)

    denominator = tf.sqrt(
        tf.reduce_mean(tf.square(true_centered)) *
        tf.reduce_mean(tf.square(pred_centered)) + eps
    )

    return -numerator / denominator


# =========================
# MUTUAL INFORMATION (LIGHT VERSION - STABLE)
# =========================
def mutual_information_loss(x, y, bins=16):

    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # normalize
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)
    y = (y - tf.reduce_min(y)) / (tf.reduce_max(y) - tf.reduce_min(y) + 1e-8)

    # discretization
    x_bin = tf.cast(x * (bins - 1), tf.int32)
    y_bin = tf.cast(y * (bins - 1), tf.int32)

    joint_idx = x_bin * bins + y_bin

    joint_hist = tf.cast(
        tf.math.bincount(
            joint_idx,
            minlength=bins * bins
        ),
        tf.float32
    )

    joint_hist = tf.reshape(joint_hist, (bins, bins))

    joint_prob = joint_hist / (tf.reduce_sum(joint_hist) + 1e-8)

    px = tf.reduce_sum(joint_prob, axis=1)
    py = tf.reduce_sum(joint_prob, axis=0)

    px_py = tf.tensordot(px, py, axes=0)

    mi = tf.reduce_sum(joint_prob * tf.math.log(joint_prob / (px_py + 1e-8) + 1e-8))

    return -mi


# =========================
# GRADIENT LOSS (FLOW REGULARIZATION)
# =========================
def gradient_loss(flow):

    dz = tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :])
    dy = tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])

    return tf.reduce_mean(dz) + tf.reduce_mean(dy) + tf.reduce_mean(dx)


# =========================
# TOTAL LOSS
# =========================
def total_loss(fixed, warped, flow, mode="ncc"):

    if mode == "ncc":
        sim_loss = ncc_loss(fixed, warped)

    elif mode == "mi":
        sim_loss = mutual_information_loss(fixed, warped)

    else:
        raise ValueError("mode must be 'ncc' or 'mi'")

    reg_loss = gradient_loss(flow)

    return sim_loss + LAMBDA_REG * reg_loss