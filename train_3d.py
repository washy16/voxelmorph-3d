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
    train_ds = train_ds.shuffle(20).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_f, val_m))
    val_ds = val_ds.batch(BATCH_SIZE)

    model = build_voxelmorph_3d()
    optimizer = tf.keras.optimizers.Adam(LR)

    for epoch in range(EPOCHS):

        print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

        train_loss = 0
        flow_mag = 0
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
    print("flow min:", tf.reduce_min(flow).numpy(),
          "flow max:", tf.reduce_max(flow).numpy())
    model.save("model_test.keras")
    print("💾 MODEL SAVED")


if __name__ == "__main__":
    train()