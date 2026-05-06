import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import RAW_T1, RAW_T2, MAX_SAMPLES

TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD NIFTI (SAFE + CANONICAL)
# =========================
def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # 🔥 uniform orientation
    data = img.get_fdata()
    data = np.nan_to_num(data)
    return data.astype(np.float32)

# =========================
# NORMALIZATION
# =========================
def normalize(img):
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)
    return (img - min_val) / (max_val - min_val + 1e-8)

# =========================
# ALIGNMENT (RESAMPLE FIX GRID)
# =========================
def resample(img):
    factors = (
        TARGET_SHAPE[0] / img.shape[0],
        TARGET_SHAPE[1] / img.shape[1],
        TARGET_SHAPE[2] / img.shape[2],
    )
    return zoom(img, factors, order=1)

# =========================
# FILE LOADER
# =========================
def get_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# =========================
# PROCESS PIPELINE
# =========================
def process():

    print("📦 PREPROCESS STARTED (ALIGNMENT DEBUG PIPELINE)")

    t1_files = get_files(RAW_T1)
    t2_files = get_files(RAW_T2)

    print(f"🔍 T1: {len(t1_files)} | T2: {len(t2_files)}")

    assert len(t1_files) > 0, "❌ No T1 files"
    assert len(t2_files) > 0, "❌ No T2 files"

    n = min(len(t1_files), len(t2_files))
    t1_files = t1_files[:n][:MAX_SAMPLES]
    t2_files = t2_files[:n][:MAX_SAMPLES]

    print(f"🔗 matched pairs: {len(t1_files)}")

    fixed_all = []
    moving_all = []

    mismatch_count = 0

    for i, (t1_path, t2_path) in enumerate(zip(t1_files, t2_files)):

        try:
            # =========================
            # LOAD
            # =========================
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            # =========================
            # 🔥 BEFORE ALIGNMENT CHECK
            # =========================
            if t1.shape != t2.shape:
                mismatch_count += 1
                print(f"⚠️ BEFORE ALIGNMENT {i}: T1={t1.shape} | T2={t2.shape}")

            # =========================
            # NORMALIZATION
            # =========================
            t1 = normalize(t1)
            t2 = normalize(t2)

            # =========================
            # ALIGNMENT (FORCED GRID)
            # =========================
            t1 = resample(t1)
            t2 = resample(t2)

            # =========================
            # 🔥 AFTER ALIGNMENT CHECK
            # =========================
            if t1.shape != t2.shape:
                print(f"❌ STILL MISMATCH AFTER ALIGNMENT {i}")
            else:
                print(f"✔ AFTER ALIGNMENT {i}: {t1.shape} | {t2.shape}")

            # =========================
            # CHANNEL DIM
            # =========================
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            fixed_all.append(t1)
            moving_all.append(t2)

            print(f"✔ processed {i+1}/{len(t1_files)}")

        except Exception as e:
            print(f"❌ error {i}: {e}")

    # =========================
    # TO ARRAY
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

    # =========================
    # SAVE DATASET
    # =========================
    os.makedirs("data", exist_ok=True)

    np.savez_compressed(
        "data/train.npz",
        fixed=fixed_all[train_idx],
        moving=moving_all[train_idx]
    )

    np.savez_compressed(
        "data/val.npz",
        fixed=fixed_all[val_idx],
        moving=moving_all[val_idx]
    )

    # =========================
    # FINAL REPORT
    # =========================
    print("\n✅ PREPROCESS DONE")
    print(f"⚠️ total mismatches BEFORE alignment: {mismatch_count}")
    print("TRAIN:", fixed_all[train_idx].shape)
    print("VAL:", fixed_all[val_idx].shape)

    print("\n🧠 READY FOR VOXELMORPH TRAINING")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process()