import tensorflow as tf
from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss

# =========================
# CONFIG
# =========================
EPOCHS = 30
BATCH_SIZE = 2
LR = 1e-4

# =========================
# CHECKPOINT PATH
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

        # 🔥 sécurité NaN
        tf.debugging.check_numerics(loss, "Loss is NaN or Inf")

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, flow


# =========================
# TRAIN FUNCTION
# =========================
def train():

    print("🚀 LOADING DATA...")
    train_fixed, train_moving, val_fixed, val_moving = load_data()

    print(f"✔ TRAIN SIZE: {len(train_fixed)}")
    print(f"✔ VAL SIZE: {len(val_fixed)}")

    # =========================
    # DATASET
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices((train_fixed, train_moving))
    train_ds = (
        train_ds
        .shuffle(buffer_size=len(train_fixed))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((val_fixed, val_moving))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # =========================
    # MODEL + OPTIMIZER
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
        steps = 0
        flow_mean_epoch = 0.0

        for fixed, moving in train_ds:
            loss, flow = train_step(model, optimizer, fixed, moving)

            train_loss += loss.numpy()
            flow_mean_epoch += tf.reduce_mean(tf.abs(flow)).numpy()
            steps += 1

        train_loss /= steps
        flow_mean_epoch /= steps

        print(f"📉 TRAIN LOSS: {train_loss:.5f}")
        print(f"🌊 FLOW MAG: {flow_mean_epoch:.6f}")

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
        print(f"📊 VAL LOSS: {val_loss:.5f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_loss < best_val:
            best_val = val_loss
            model.save("best_voxelmorph_3d.h5")
            print("🏆 BEST MODEL SAVED")

        # =========================
        # CHECKPOINT
        # =========================
        manager.save()
        print("💾 CHECKPOINT SAVED")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()