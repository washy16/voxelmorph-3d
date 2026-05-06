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
        tf.reduce_mean(tf.square(true_centered)) *
        tf.reduce_mean(tf.square(pred_centered)) + eps
    )

    return -numerator / denominator


# =========================
# MUTUAL INFORMATION (STABLE VERSION)
# =========================
def mutual_information_loss(fixed, warped, num_bins=16, eps=1e-6):

    fixed = tf.reshape(fixed, [-1])
    warped = tf.reshape(warped, [-1])

    # normalize [0,1]
    fixed = (fixed - tf.reduce_min(fixed)) / (tf.reduce_max(fixed) - tf.reduce_min(fixed) + eps)
    warped = (warped - tf.reduce_min(warped)) / (tf.reduce_max(warped) - tf.reduce_min(warped) + eps)

    bins = tf.linspace(0.0, 1.0, num_bins)
    sigma = 0.1

    def soft_hist(x):
        x = tf.expand_dims(x, -1)
        b = tf.expand_dims(bins, 0)
        weights = tf.exp(-((x - b) ** 2) / (2 * sigma ** 2))
        return tf.reduce_mean(weights, axis=0)

    p_f = soft_hist(fixed)
    p_w = soft_hist(warped)

    joint = tf.einsum('i,j->ij', p_f, p_w)
    joint = joint / (tf.reduce_sum(joint) + eps)

    p_f = p_f / (tf.reduce_sum(p_f) + eps)
    p_w = p_w / (tf.reduce_sum(p_w) + eps)

    H_f = -tf.reduce_sum(p_f * tf.math.log(p_f + eps))
    H_w = -tf.reduce_sum(p_w * tf.math.log(p_w + eps))
    H_fw = -tf.reduce_sum(joint * tf.math.log(joint + eps))

    mi = H_f + H_w - H_fw

    return -mi


# =========================
# FLOW SMOOTHNESS LOSS (FIXED)
# =========================
def gradient_loss(flow):

    # spatial differences (3D)
    dx = tf.reduce_mean(tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]))
    dy = tf.reduce_mean(tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]))
    dz = tf.reduce_mean(tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :]))

    return dx + dy + dz


# =========================
# TOTAL LOSS
# =========================
def total_loss(fixed, warped, flow, mode="ncc"):

    # similarity term
    if mode == "ncc":
        sim_loss = ncc_loss(fixed, warped)

    elif mode == "mi":
        sim_loss = mutual_information_loss(fixed, warped)

    else:
        raise ValueError("mode must be 'ncc' or 'mi'")

    # regularization
    reg_loss = gradient_loss(flow)

    return sim_loss + LAMBDA_REG * reg_loss