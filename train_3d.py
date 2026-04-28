import tensorflow as tf
from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss

# =========================
# CONFIG
# =========================
EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-4

# =========================
# CHECKPOINT PATH (LAB MODE)
# =========================
checkpoint_dir = "/content/drive/MyDrive/voxelmorph-checkpoints"

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
# TRAIN FUNCTION
# =========================
def train():

    print("🚀 LOADING DATA...")
    train_fixed, train_moving, val_fixed, val_moving = load_data()

    # DEBUG LIMIT (safe training)
    train_fixed = train_fixed[:10]
    train_moving = train_moving[:10]
    val_fixed = val_fixed[:5]
    val_moving = val_moving[:5]

    print(f"✔ TRAIN SIZE: {len(train_fixed)}")
    print(f"✔ VAL SIZE: {len(val_fixed)}")

    # =========================
    # DATASET
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices((train_fixed, train_moving))
    train_ds = train_ds.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_fixed, val_moving))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # =========================
    # MODEL + OPTIMIZER
    # =========================
    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    # =========================
    # CHECKPOINT SYSTEM (LAB)
    # =========================
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    # restore if exists
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("🔁 Restored from checkpoint:", manager.latest_checkpoint)
    else:
        print("🆕 Starting fresh training")

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
        steps = 0

        for fixed, moving in train_ds:
            loss = train_step(model, optimizer, fixed, moving)
            train_loss += loss
            steps += 1

        train_loss /= float(steps)
        print(f"📉 TRAIN LOSS: {train_loss.numpy():.5f}")

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
            print("🏆 BEST MODEL SAVED")

        # =========================
        # CHECKPOINT SAVE (LAB)
        # =========================
        manager.save()
        print("💾 CHECKPOINT SAVED")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()