import os
import tensorflow as tf
from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss


# =========================
# CONFIG
# =========================
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4


# 🔥 GPU memory fix (important sur Colab)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# =========================
# DATA GENERATOR
# =========================
def data_generator(fixed, moving):
    for f, m in zip(fixed, moving):
        yield f, m


# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:
        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# =========================
# MAIN TRAIN LOOP
# =========================
def train():

    print("🚀 LOADING DATA...")
    train_fixed, train_moving, val_fixed, val_moving = load_data()

    # 🔥 small subset for stability
    train_fixed = train_fixed[:50]
    train_moving = train_moving[:50]
    val_fixed = val_fixed[:10]
    val_moving = val_moving[:10]

    print(f"✔ TRAIN SIZE: {len(train_fixed)}")
    print(f"✔ VAL SIZE: {len(val_fixed)}")


    # =========================
    # TF DATASET (SAFE VERSION)
    # =========================
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_fixed, train_moving),
        output_signature=(
            tf.TensorSpec(shape=(96, 96, 96, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(96, 96, 96, 1), dtype=tf.float32),
        )
    ).batch(BATCH_SIZE)


    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_fixed, val_moving),
        output_signature=(
            tf.TensorSpec(shape=(96, 96, 96, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(96, 96, 96, 1), dtype=tf.float32),
        )
    ).batch(BATCH_SIZE)


    # =========================
    # MODEL
    # =========================
    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    best_val = float("inf")


    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        # -------- TRAIN --------
        train_loss = 0.0
        steps = 0

        for fixed, moving in train_ds:
            loss = train_step(model, optimizer, fixed, moving)
            train_loss += loss
            steps += 1

        train_loss /= steps
        print(f"📉 TRAIN LOSS: {train_loss.numpy():.5f}")


        # -------- VALIDATION --------
        val_loss = 0.0
        vsteps = 0

        for fixed, moving in val_ds:
            warped, flow = model([moving, fixed], training=False)
            loss = total_loss(fixed, warped, flow)
            val_loss += loss
            vsteps += 1

        val_loss /= vsteps
        print(f"📊 VAL LOSS: {val_loss.numpy():.5f}")


        # -------- SAVE BEST --------
        if val_loss < best_val:
            best_val = val_loss
            model.save("best_voxelmorph_3d.h5")
            print("🏆 SAVED BEST MODEL")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()