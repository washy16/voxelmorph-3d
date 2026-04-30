import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
RAW_T1 = "/content/drive/MyDrive/voxelmorph-3d/data/raw/T1"
RAW_T2 = "/content/drive/MyDrive/voxelmorph-3d/data/raw/T2"

CACHE_T1 = "/content/drive/MyDrive/voxelmorph-3d/data/cache/T1"
CACHE_T2 = "/content/drive/MyDrive/voxelmorph-3d/data/cache/T2"

os.makedirs(CACHE_T1, exist_ok=True)
os.makedirs(CACHE_T2, exist_ok=True)

TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD + ORIENTATION FIX
# =========================
def load_nifti(path):
    img = nib.load(path)

    # 🔥 CRUCIAL : orientation standard (RAS)
    img = nib.as_closest_canonical(img)

    data = img.get_fdata()
    data = np.nan_to_num(data)
    data = data.astype(np.float32)

    return data

# =========================
# NORMALIZATION
# =========================
def normalize(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-8)

    # 🔥 stabilisation
    img = np.clip(img, -5, 5)

    return img

# =========================
# TRUE 3D RESIZE
# =========================
def resize_volume(img, target_shape=TARGET_SHAPE):

    factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    )

    img = zoom(img, factors, order=1)  # trilinear

    return img

# =========================
# MATCH T1 / T2
# =========================
def get_pairs():

    t1_files = sorted([f for f in os.listdir(RAW_T1) if "T1" in f])
    t2_files = sorted([f for f in os.listdir(RAW_T2) if "T2" in f])

    t2_dict = {}

    # index T2 par ID
    for t2 in t2_files:
        base = t2.replace("-T2.nii.gz", "").replace("_T2.nii.gz", "").replace(".nii.gz", "")
        t2_dict[base] = t2

    pairs = []

    for t1 in t1_files:

        base = t1.replace("-T1.nii.gz", "").replace("_T1.nii.gz", "").replace(".nii.gz", "")

        if base in t2_dict:

            pairs.append((
                os.path.join(RAW_T1, t1),
                os.path.join(RAW_T2, t2_dict[base])
            ))

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
            # LOAD + FIX ORIENTATION
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            # NORMALIZE
            t1 = normalize(t1)
            t2 = normalize(t2)

            # TRUE 3D RESIZE
            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            # ADD CHANNEL
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            # SAVE CACHE (optionnel)
            base = os.path.basename(t1_path).replace(".nii.gz", "").replace(".nii", "")

            np.savez_compressed(os.path.join(CACHE_T1, base + ".npz"), img=t1)
            np.savez_compressed(os.path.join(CACHE_T2, base + ".npz"), img=t2)

            fixed_all.append(t1)
            moving_all.append(t2)

            print(f"✔ processed {i+1}/{len(pairs)}")

        except Exception as e:
            print(f"❌ ERROR on {t1_path}: {e}")
            continue

    # =========================
    # CONVERT TO ARRAY
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
    np.savez_compressed("data/train.npz", fixed=train_fixed, moving=train_moving)
    np.savez_compressed("data/val.npz", fixed=val_fixed, moving=val_moving)

    print("\n✅ PROCESS COMPLETED")
    print("TRAIN:", train_fixed.shape)
    print("VAL:", val_fixed.shape)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    process()