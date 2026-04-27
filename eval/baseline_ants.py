import ants
import numpy as np


def ants_register(fixed_path, moving_path):
    """
    Baseline registration ANTs SyN (state-of-the-art classical method)
    """

    # Load images
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    # 🔥 Symmetric normalization (SyN)
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN"
    )

    warped = registration["warpedmovout"]

    return warped, registration


def compute_similarity(fixed, warped):
    """
    Simple similarity metric (NCC-like proxy)
    """

    f = fixed.numpy().flatten()
    w = warped.numpy().flatten()

    f = (f - np.mean(f)) / (np.std(f) + 1e-8)
    w = (w - np.mean(w)) / (np.std(w) + 1e-8)

    return np.mean(f * w)


def run_baseline(fixed_path, moving_path):

    warped, reg = ants_register(fixed_path, moving_path)

    sim = compute_similarity(
        ants.image_read(fixed_path),
        warped
    )

    print("🧠 ANTs baseline similarity:", sim)

    return warped, sim