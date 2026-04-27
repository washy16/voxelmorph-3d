import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
RAW_T1 = "/content/drive/MyDrive/VoxelMorph/data/raw/T1"
RAW_T2 = "/content/drive/MyDrive/VoxelMorph/data/raw/T2"

CACHE_T1 = "data/cache/T1"
CACHE_T2 = "data/cache/T2"

os.makedirs(CACHE_T1, exist_ok=True)
os.makedirs(CACHE_T2, exist_ok=True)

TARGET_SHAPE = (96, 96, 96) # ⚠️ tu peux mettre (96,96,96) si lent


# =========================
# LOAD NIFTI
# =========================
def load_nifti(path):
    img = nib.load(path).get_fdata()
    img = np.nan_to_num(img)
    img = img.astype(np.float32)
    return img


# =========================
# NORMALIZATION
# =========================
def normalize(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-8)
    return img


# =========================
# RESIZE 3D (STABLE)
# =========================
def resize_volume(img, target_shape=TARGET_SHAPE):

    # resize XY
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, target_shape[:2])

    # resize Z
    img = tf.transpose(img, [2, 0, 1])  # (Z, X, Y)
    img = tf.image.resize(img, (target_shape[2], target_shape[0]))
    img = tf.transpose(img, [1, 2, 0])

    return img.numpy()


# =========================
# MATCH T1 / T2
# =========================
def get_pairs():

    t1_files = sorted([
        f for f in os.listdir(RAW_T1)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

    t2_files = sorted([
        f for f in os.listdir(RAW_T2)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

    pairs = []

    for t1 in t1_files:
        base = t1.split("-T1")[0]

        for t2 in t2_files:
            if base in t2:
                pairs.append((
                    os.path.join(RAW_T1, t1),
                    os.path.join(RAW_T2, t2)
                ))
                break

    return pairs


# =========================
# MAIN PROCESS
# =========================
def process():

    pairs = get_pairs()
    print(f"🔍 FOUND PAIRS: {len(pairs)}")

    fixed_all = []
    moving_all = []

    for i, (t1_path, t2_path) in enumerate(pairs):

        try:
            # LOAD
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            # NORMALIZE
            t1 = normalize(t1)
            t2 = normalize(t2)

            # RESIZE (CRUCIAL)
            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            # ADD CHANNEL
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            # SAVE CACHE
            base = os.path.basename(t1_path).replace(".nii.gz", "").replace(".nii", "")

            np.savez_compressed(
                os.path.join(CACHE_T1, base + ".npz"),
                img=t1
            )

            np.savez_compressed(
                os.path.join(CACHE_T2, base + ".npz"),
                img=t2
            )

            fixed_all.append(t1)
            moving_all.append(t2)

            print(f"✔ processed {i+1}/{len(pairs)}")

        except Exception as e:
            print(f"❌ ERROR on {t1_path}: {e}")
            continue

    # =========================
    # CONVERT TO ARRAY (NOW SAFE)
    # =========================
    fixed_all = np.array(fixed_all, dtype=np.float32)
    moving_all = np.array(moving_all, dtype=np.float32)

    # =========================
    # TRAIN / VAL SPLIT
    # =========================
    idx = np.arange(len(fixed_all))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_fixed = fixed_all[train_idx]
    train_moving = moving_all[train_idx]

    val_fixed = fixed_all[val_idx]
    val_moving = moving_all[val_idx]

    # =========================
    # SAVE FINAL DATASETS
    # =========================
    np.savez_compressed(
        "data/train.npz",
        fixed=train_fixed,
        moving=train_moving
    )

    np.savez_compressed(
        "data/val.npz",
        fixed=val_fixed,
        moving=val_moving
    )

    print("\n✅ PROCESS COMPLETED")
    print("TRAIN:", train_fixed.shape)
    print("VAL:", val_fixed.shape)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process()