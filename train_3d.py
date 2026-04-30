import tensorflow as tf
from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss

# =========================
# CONFIG
# =========================
EPOCHS = 30
BATCH_SIZE = 1   # 🔥 IMPORTANT (3D = 1)
LR = 1e-4

checkpoint_dir = "/content/drive/MyDrive/voxelmorph-checkpoints"

# =========================
# TRAIN STEP (OPTIMISÉ GPU)
# =========================
@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:
        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow)

    tf.debugging.check_numerics(loss, "NaN detected")

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, flow


# =========================
# TRAIN
# =========================
def train():

    print("🚀 LOADING DATA...")
    train_fixed, train_moving, val_fixed, val_moving = load_data()

    # 🔥 LIMITATION STABLE GPU
    train_fixed = train_fixed[:80]
    train_moving = train_moving[:80]
    val_fixed = val_fixed[:20]
    val_moving = val_moving[:20]

    print("✔ TRAIN:", train_fixed.shape)
    print("✔ VAL:", val_fixed.shape)

    # =========================
    # DATASET (OPTIMISÉ)
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices((train_fixed, train_moving))
    train_ds = train_ds.shuffle(20).batch(BATCH_SIZE).prefetch(1)

    val_ds = tf.data.Dataset.from_tensor_slices((val_fixed, val_moving))
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
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("🔁 Restored:", manager.latest_checkpoint)

    best_val = 1e9

    # =========================
    # EPOCH LOOP
    # =========================
    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        train_loss = 0.0
        val_loss = 0.0
        steps = 0

        # =========================
        # TRAIN
        # =========================
        for fixed, moving in train_ds:
            loss, flow = train_step(model, optimizer, fixed, moving)
            train_loss += loss
            steps += 1

        train_loss = train_loss / steps

        # FLOW (simple & stable)
        flow_mean = tf.reduce_mean(tf.abs(flow))

        print("📉 TRAIN LOSS:", train_loss.numpy())
        print("🌊 FLOW:", flow_mean.numpy())

        # =========================
        # VALIDATION
        # =========================
        val_steps = 0

        for fixed, moving in val_ds:
            warped, flow = model([moving, fixed], training=False)
            loss = total_loss(fixed, warped, flow)

            val_loss += loss
            val_steps += 1

        val_loss = val_loss / val_steps

        print("📊 VAL LOSS:", val_loss.numpy())

        # =========================
        # SAVE BEST
        # =========================
        if val_loss < best_val:
            best_val = val_loss
            model.save("best_voxelmorph_3d.h5")
            print("🏆 BEST MODEL SAVED")

        manager.save()
        print("💾 CHECKPOINT SAVED")


# RUN
if __name__ == "__main__":
    train()