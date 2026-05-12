import tensorflow as tf
import numpy as np

from model_3d import build_voxelmorph_3d
from data_loader_3d import load_data
from losses import total_loss
from config import LR, EPOCHS, BATCH_SIZE


# =========================
# TRAIN STEP
# =========================
@tf.function
def train_step(model, optimizer, fixed, moving):

    with tf.GradientTape() as tape:

        # forward
        warped, flow = model([moving, fixed], training=True)

        # total loss (MI + regularization)
        loss = total_loss(fixed, warped, flow)

    # gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # apply gradients
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables)
    )

    return loss, flow, warped


# =========================
# TRAINING LOOP
# =========================
def train():

    print("🚀 LOADING DATA")

    train_f, train_m, val_f, val_m = load_data()

    print("✔ TRAIN FIXED:", train_f.shape)
    print("✔ TRAIN MOVING:", train_m.shape)

    print("✔ VAL FIXED:", val_f.shape)
    print("✔ VAL MOVING:", val_m.shape)

    # =========================
    # DATASET TF
    # =========================
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_f, train_m)
    )

    train_ds = train_ds.batch(BATCH_SIZE)

    # =========================
    # MODEL
    # =========================
    print("\n🧠 BUILDING VOXELMORPH")

    model = build_voxelmorph_3d()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR
    )

    print("✔ MODEL READY")

    # =========================
    # TRAINING
    # =========================
    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch + 1}/{EPOCHS}")

        epoch_loss = []
        epoch_flow = []

        # -------------------------
        # iterate batches
        # -------------------------
        for fixed, moving in train_ds:

            loss, flow, warped = train_step(
                model,
                optimizer,
                fixed,
                moving
            )

            epoch_loss.append(loss.numpy())

            epoch_flow.append(
                tf.reduce_mean(
                    tf.abs(flow)
                ).numpy()
            )

        # =========================
        # METRICS
        # =========================
        mean_loss = np.mean(epoch_loss)
        mean_flow = np.mean(epoch_flow)

        print(f"📉 LOSS: {mean_loss:.6f}")
        print(f"🌊 FLOW: {mean_flow:.6f}")

        # optional debug
        print(
            "flow min:",
            tf.reduce_min(flow).numpy(),
            "flow max:",
            tf.reduce_max(flow).numpy()
        )

    # =========================
    # SAVE MODEL
    # =========================
    print("\n💾 SAVING MODEL")

    model.save("model_test.keras")

    print("✅ TRAINING COMPLETE")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()