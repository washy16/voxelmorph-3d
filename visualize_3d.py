# visualize_3d.py

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# =========================
# PREPARE VOLUME
# =========================
def prepare_volume(volume):

    volume = np.array(volume)

    # (1, H, W, D, 1)
    if volume.ndim == 5:
        volume = volume[0, :, :, :, 0]

    # (H, W, D, 1)
    elif volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[:, :, :, 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Volume must be 3D after processing. Got shape: {volume.shape}"
        )

    return volume


# =========================
# SHOW SINGLE SLICE
# =========================
def show_slices(volume, title="Volume", axis=2):

    volume = prepare_volume(volume)

    # Slice centrale
    mid = volume.shape[axis] // 2

    if axis == 0:
        img = volume[mid, :, :]

    elif axis == 1:
        img = volume[:, mid, :]

    else:
        img = volume[:, :, mid]

    plt.figure(figsize=(6, 6))

    plt.imshow(
        img.T,
        cmap="gray",
        origin="lower"
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show(block=True)


# =========================
# COMPARE 3 VOLUMES
# =========================
def compare_triplet(fixed, moving, warped, axis=2):

    print("\n🧠 VISUALIZATION START\n")

    show_slices(fixed, "Fixed", axis)
    show_slices(moving, "Moving", axis)
    show_slices(warped, "Warped (VoxelMorph)", axis)


# =========================
# SHOW FLOW MAGNITUDE
# =========================
def show_flow(flow):

    flow = np.array(flow)

    # (1, H, W, D, 3)
    if flow.ndim == 5:
        flow = flow[0]

    if flow.ndim != 4:
        raise ValueError(
            f"Flow must be 4D. Got shape: {flow.shape}"
        )

    # Magnitude du déplacement
    magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))

    mid = magnitude.shape[2] // 2

    plt.figure(figsize=(6, 6))

    plt.imshow(
        magnitude[:, :, mid].T,
        cmap="jet",
        origin="lower"
    )

    plt.title("Flow Magnitude")
    plt.axis("off")

    plt.colorbar()

    plt.tight_layout()
    plt.show(block=True)


# =========================
# QUICK TEST
# =========================
if __name__ == "__main__":

    print("✅ TEST VISUALIZE_3D")

    # Faux volume 3D
    x = np.random.rand(96, 96, 96)

    # Faux flow
    flow = np.random.rand(96, 96, 96, 3)

    show_slices(x, "Test Volume")

    show_flow(flow)

    print("✅ FINISHED")