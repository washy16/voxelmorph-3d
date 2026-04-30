import tensorflow as tf

from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss
from config import LR, EPOCHS, BATCH_SIZE, CKPT_PATH


# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:
        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow)

        # 🔥 sécurité
        tf.debugging.check_numerics(loss, "NaN detected")

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, flow


# =========================
# TRAIN
# =========================
def train():

    print("🚀 LOADING DATA")
    train_f, train_m, val_f, val_m = load_data()

    # 🔥 DEBUG MODE (IMPORTANT)
    train_f = train_f[:20]
    train_m = train_m[:20]
    val_f = val_f[:5]
    val_m = val_m[:5]

    print("✔ TRAIN:", len(train_f))
    print("✔ VAL:", len(val_f))

    # =========================
    # DATASET
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices((train_f, train_m))
    train_ds = train_ds.shuffle(20).batch(BATCH_SIZE).prefetch(1)

    val_ds = tf.data.Dataset.from_tensor_slices((val_f, val_m))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)

    # =========================
    # MODEL
    # =========================
    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    # =========================
    # CHECKPOINT
    # =========================
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("🔁 Restored:", manager.latest_checkpoint)
    else:
        print("🆕 New training")

    best_val = float("inf")

    # =========================
    # EPOCH LOOP
    # =========================
    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        # =========================
        # TRAIN
        # =========================
        train_loss = 0.0
        flow_mag = 0.0
        steps = 0

        for fixed, moving in train_ds:
            loss, flow = train_step(model, optimizer, fixed, moving)

            train_loss += loss.numpy()
            flow_mag += tf.reduce_mean(tf.abs(flow)).numpy()
            steps += 1

        train_loss /= steps
        flow_mag /= steps

        print(f"📉 TRAIN: {train_loss:.5f}")
        print(f"🌊 FLOW: {flow_mag:.6f}")

        # =========================
        # VALIDATION
        # =========================
        val_loss = 0.0
        val_steps = 0

        for fixed, moving in val_ds:
            warped, flow = model([moving, fixed], training=False)
            loss = total_loss(fixed, warped, flow)

            val_loss += loss.numpy()
            val_steps += 1

        val_loss /= val_steps

        print(f"📊 VAL: {val_loss:.5f}")

        # =========================
        # SAVE BEST
        # =========================
        if val_loss < best_val:
            best_val = val_loss
            model.save("best_voxelmorph_3d.keras")
            print("🏆 BEST MODEL SAVED")

        manager.save()
        print("💾 CHECKPOINT SAVED")


if __name__ == "__main__":
    train()