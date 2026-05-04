import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import RAW_T1, RAW_T2, MAX_SAMPLES

TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD NIFTI + ORIENTATION FIX
# =========================
def load_nifti(path):
    img = nib.load(path)

    # 🔥 IMPORTANT : standardisation orientation
    img = nib.as_closest_canonical(img)

    data = img.get_fdata()
    data = np.nan_to_num(data)

    return data.astype(np.float32)

# =========================
# NORMALIZATION (ROBUST)
# =========================
def normalize(img):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    return (img - p1) / (p99 - p1 + 1e-8)

# =========================
# RESIZE TO COMMON GRID
# =========================
def resize_volume(img):
    factors = (
        TARGET_SHAPE[0] / img.shape[0],
        TARGET_SHAPE[1] / img.shape[1],
        TARGET_SHAPE[2] / img.shape[2],
    )
    return zoom(img, factors, order=1)

# =========================
# GET FILES
# =========================
def get_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# =========================
# SAFE PAIRING (IMPORTANT)
# =========================
def get_id(filename):
    # ex: IXI123-xxx-T1.nii.gz → IXI123
    return os.path.basename(filename).split("-")[0]


def build_pairs(t1_files, t2_files):
    t1_dict = {get_id(f): f for f in t1_files}
    t2_dict = {get_id(f): f for f in t2_files}

    common_ids = sorted(set(t1_dict.keys()) & set(t2_dict.keys()))

    pairs = [(t1_dict[i], t2_dict[i]) for i in common_ids]

    return pairs

# =========================
# MAIN PROCESS
# =========================
def process():

    print("📦 PREPROCESS STARTED (CLEAN PIPELINE)")

    t1_files = get_files(RAW_T1)
    t2_files = get_files(RAW_T2)

    print(f"🔍 T1: {len(t1_files)} | T2: {len(t2_files)}")

    assert len(t1_files) > 0, "No T1 files found"
    assert len(t2_files) > 0, "No T2 files found"

    # 🔥 build correct subject pairs
    pairs = build_pairs(t1_files, t2_files)

    print(f"🔗 matched pairs: {len(pairs)}")

    pairs = pairs[:MAX_SAMPLES]

    fixed_all = []
    moving_all = []

    for i, (t1_path, t2_path) in enumerate(pairs):

        try:
            # LOAD
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            # SHAPE CHECK
            if t1.shape != t2.shape:
                print(f"⚠️ shape mismatch {i}: {t1.shape} vs {t2.shape}")

            # NORMALIZE
            t1 = normalize(t1)
            t2 = normalize(t2)

            # RESIZE (common grid)
            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            # ADD CHANNEL DIM
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            # STORE
            fixed_all.append(t1)   # T1 reference
            moving_all.append(t2)  # T2 deformable

            print(f"✔ processed {i+1}/{len(pairs)}")

        except Exception as e:
            print(f"❌ error {i}: {e}")

    # =========================
    # TO ARRAY
    # =========================
    fixed_all = np.array(fixed_all, dtype=np.float32)
    moving_all = np.array(moving_all, dtype=np.float32)

    # =========================
    # SPLIT TRAIN / VAL
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
    # DEBUG
    # =========================
    print("\n✅ PREPROCESS DONE")
    print("TRAIN:", fixed_all[train_idx].shape)
    print("VAL:", fixed_all[val_idx].shape)

    print("\n🧠 READY FOR VOXELMORPH TRAINING")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process()