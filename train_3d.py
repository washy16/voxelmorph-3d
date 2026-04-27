import tensorflow as tf
from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss


# =========================
# CONFIG
# =========================
EPOCHS = 20
BATCH_SIZE = 2   # 🔥 augmente à 4 si ça passe
LR = 1e-4


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

    print(f"✔ TRAIN SIZE: {len(train_fixed)}")
    print(f"✔ VAL SIZE: {len(val_fixed)}")

    # =========================
    # DATASET PIPELINE
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices((train_fixed, train_moving))
    train_ds = train_ds.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_fixed, val_moving))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # =========================
    # MODEL
    # =========================
    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    best_val = float("inf")

    # =========================
    # EPOCH LOOP
    # =========================
    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        # =========================
        # TRAIN
        # =========================
        epoch_loss = 0.0
        steps = 0

        for fixed, moving in train_ds:

            loss = train_step(model, optimizer, fixed, moving)

            epoch_loss += loss
            steps += 1

        epoch_loss /= float(steps)

        print(f"📉 TRAIN LOSS: {epoch_loss.numpy():.5f}")

        # =========================
        # VALIDATION
        # =========================
        val_loss = 0.0
        val_steps = 0

        for fixed, moving in val_ds:

            warped, flow = model([moving, fixed], training=False)
            loss = total_loss(fixed, warped, flow)

            val_loss += loss
            val_steps += 1

        val_loss /= float(val_steps)

        print(f"📊 VAL LOSS: {val_loss.numpy():.5f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_loss < best_val:

            best_val = val_loss
            model.save("best_voxelmorph_3d.h5")

            print("🏆 SAVED BEST MODEL")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()