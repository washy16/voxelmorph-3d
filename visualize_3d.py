# visualize_3d.py

import numpy as np
import matplotlib.pyplot as plt


# =========================
# SLICE VIEWER 3D
# =========================
def show_slices(volume, title="Volume", axis=2):

    if len(volume.shape) == 5:
        volume = volume[0, :, :, :, 0]

    mid = volume.shape[axis] // 2

    if axis == 0:
        img = volume[mid, :, :]
    elif axis == 1:
        img = volume[:, mid, :]
    else:
        img = volume[:, :, mid]

    plt.figure(figsize=(5, 5))
    plt.imshow(img.T, cmap="gray", origin="lower")
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================
# COMPARE FIXED / MOVING / WARPED
# =========================
def compare_triplet(fixed, moving, warped):

    print("🧠 VISUALIZATION START")

    show_slices(fixed, "Fixed")
    show_slices(moving, "Moving")
    show_slices(warped, "Warped (VoxelMorph)")


# =========================
# FLOW VISUAL (MAGNITUDE)
# =========================
def show_flow(flow):

    if len(flow.shape) == 5:
        flow = flow[0]

    mag = np.sqrt(np.sum(flow**2, axis=-1))

    mid = mag.shape[0] // 2

    plt.figure(figsize=(5, 5))
    plt.imshow(mag[mid].T, cmap="jet", origin="lower")
    plt.title("Flow magnitude")
    plt.axis("off")
    plt.colorbar()
    plt.show()