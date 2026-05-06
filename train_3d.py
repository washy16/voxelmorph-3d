# train_3d.py

import tensorflow as tf

from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss
from config import LR, EPOCHS, BATCH_SIZE


@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:

        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, flow


def train():

    print("🚀 LOADING DATA")

    train_f, train_m, val_f, val_m = load_data()

    print("✔ TRAIN:", len(train_f))
    print("✔ VAL:", len(val_f))

    train_ds = tf.data.Dataset.from_tensor_slices((train_f, train_m))
    train_ds = train_ds.shuffle(100).repeat().batch(BATCH_SIZE)

    steps = max(1, len(train_f) // BATCH_SIZE)

    model = build_voxelmorph_3d()
    opt = tf.keras.optimizers.Adam(LR)

    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        loss_sum = 0
        flow_sum = 0

        for step, (fixed, moving) in enumerate(train_ds.take(steps)):

            loss, flow = train_step(model, opt, fixed, moving)

            loss_sum += loss.numpy()
            flow_sum += tf.reduce_mean(tf.abs(flow)).numpy()

        print("📉 LOSS:", loss_sum / steps)
        print("🌊 FLOW:", flow_sum / steps)

    model.save("model_test.keras")
    print("💾 MODEL SAVED")


if __name__ == "__main__":
    train()