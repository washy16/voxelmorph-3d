import tensorflow as tf
from config import LAMBDA_REG


# =========================
# NORMALIZED CROSS CORRELATION (NCC)
# =========================
def ncc_loss(y_true, y_pred, eps=1e-5):

    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)

    true_centered = y_true - mean_true
    pred_centered = y_pred - mean_pred

    numerator = tf.reduce_mean(true_centered * pred_centered)
    denominator = tf.sqrt(
        tf.reduce_mean(true_centered ** 2) *
        tf.reduce_mean(pred_centered ** 2) + eps
    )

    return -numerator / denominator


# =========================
# FLOW SMOOTHNESS LOSS (CRITIQUE)
# =========================
def gradient_loss(flow):

    dz = tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :])
    dy = tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])

    return tf.reduce_mean(dz) + tf.reduce_mean(dy) + tf.reduce_mean(dx)


# =========================
# TOTAL LOSS VOXELMORPH
# =========================
def total_loss(fixed, warped, flow):

    sim_loss = ncc_loss(fixed, warped)
    reg_loss = gradient_loss(flow)

    return sim_loss + LAMBDA_REG * reg_loss