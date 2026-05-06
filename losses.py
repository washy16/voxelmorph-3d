# losses.py

import tensorflow as tf
from config import LAMBDA_REG


def ncc_loss(y_true, y_pred, eps=1e-5):

    mean_t = tf.reduce_mean(y_true)
    mean_p = tf.reduce_mean(y_pred)

    t = y_true - mean_t
    p = y_pred - mean_p

    num = tf.reduce_mean(t * p)

    den = tf.sqrt(tf.reduce_mean(t**2) * tf.reduce_mean(p**2) + eps)

    return -num / den


def gradient_loss(flow):

    dx = tf.reduce_mean(tf.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]))
    dy = tf.reduce_mean(tf.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]))
    dz = tf.reduce_mean(tf.abs(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :]))

    return dx + dy + dz


def total_loss(fixed, warped, flow):

    sim = ncc_loss(fixed, warped)
    reg = gradient_loss(flow)

    return sim + LAMBDA_REG * reg