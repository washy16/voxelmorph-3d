import tensorflow as tf

from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss
from config import LR, EPOCHS, BATCH_SIZE, CKPT_PATH


@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:
        warped, flow = model([moving, fixed], training=True)
        loss = total_loss(fixed, warped, flow)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train():

    print("🚀 LOADING DATA")
    train_f, train_m, val_f, val_m = load_data()

    train_ds = tf.data.Dataset.from_tensor_slices((train_f, train_m))
    train_ds = train_ds.shuffle(50).batch(BATCH_SIZE).prefetch(1)

    val_ds = tf.data.Dataset.from_tensor_slices((val_f, val_m))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)

    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)

    best_val = float("inf")

    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        # TRAIN
        train_loss = 0
        steps = 0

        for fixed, moving in train_ds:
            loss = train_step(model, optimizer, fixed, moving)
            train_loss += loss
            steps += 1

        train_loss /= steps

        print("📉 TRAIN:", train_loss.numpy())

        # VAL
        val_loss = 0
        val_steps = 0

        for fixed, moving in val_ds:
            warped, flow = model([moving, fixed], training=False)
            loss = total_loss(fixed, warped, flow)

            val_loss += loss
            val_steps += 1

        val_loss /= val_steps

        print("📊 VAL:", val_loss.numpy())

        # SAVE BEST
        if val_loss < best_val:
            best_val = val_loss
            model.save("best_voxelmorph_3d.keras")
            print("🏆 BEST MODEL SAVED")

        manager.save()
        print("💾 CHECKPOINT SAVED")


if __name__ == "__main__":
    train()