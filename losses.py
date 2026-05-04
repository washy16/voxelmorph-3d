import tensorflow as tf
from config import LAMBDA_REG

# =========================
# LOCAL NORMALIZED CROSS CORRELATION (STABLE VERSION)
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
# FLOW SMOOTHNESS LOSS (GRADIENT PENALTY)
# =========================
def gradient_loss(flow):

    dz = tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :])
    dy = tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])

    return (
        tf.reduce_mean(dz) +
        tf.reduce_mean(dy) +
        tf.reduce_mean(dx)
    )


# =========================
# TOTAL LOSS (VOXELMORPH STYLE)
# =========================
def total_loss(fixed, warped, flow):

    # similarity term (alignment)
    sim_loss = ncc_loss(fixed, warped)

    # regularization (smooth deformation field)
    reg_loss = gradient_loss(flow)

    return sim_loss + LAMBDA_REG * reg_loss