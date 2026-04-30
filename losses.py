import tensorflow as tf
from config import LAMBDA_REG


def ncc_loss(x, y, eps=1e-5):

    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)

    xm = x - mean_x
    ym = y - mean_y

    num = tf.reduce_mean(xm * ym)
    denom = tf.sqrt(tf.reduce_mean(xm**2) * tf.reduce_mean(ym**2) + eps)

    return -num / denom


def gradient_loss(flow):

    dy = tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :])
    dx = tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dz = tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])

    dy = tf.pad(dy, [[0,0],[0,1],[0,0],[0,0],[0,0]])
    dx = tf.pad(dx, [[0,0],[0,0],[0,1],[0,0],[0,0]])
    dz = tf.pad(dz, [[0,0],[0,0],[0,0],[0,1],[0,0]])

    return tf.reduce_mean(dx + dy + dz)


def total_loss(fixed, warped, flow):

    sim = ncc_loss(fixed, warped)
    smooth = gradient_loss(flow)

    return sim + LAMBDA_REG * smooth