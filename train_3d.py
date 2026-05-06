import tensorflow as tf

from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss
from config import LR, BATCH_SIZE


# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:
        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow, mode="ncc")  # 🔥 START NCC

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, flow


# =========================
# TRAIN LOOP
# =========================
def train():

    print("🚀 LOADING DATA")
    train_f, train_m, val_f, val_m = load_data()

    print("✔ TRAIN:", len(train_f))
    print("✔ VAL:", len(val_f))

    train_ds = tf.data.Dataset.from_tensor_slices((train_f, train_m))
    train_ds = train_ds.shuffle(10).batch(BATCH_SIZE)

    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    EPOCHS = 5   # 🔥 FAST DEBUG MODE

    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        total_loss_val = 0
        flow_mag = 0
        steps = 0

        for fixed, moving in train_ds:

            loss, flow = train_step(model, optimizer, fixed, moving)

            total_loss_val += loss.numpy()
            flow_mag += tf.reduce_mean(tf.abs(flow)).numpy()
            steps += 1

        print("📉 LOSS:", total_loss_val / steps)
        print("🌊 FLOW:", flow_mag / steps)

    model.save("model_test.keras")
    print("💾 MODEL SAVED")


if __name__ == "__main__":
    train()