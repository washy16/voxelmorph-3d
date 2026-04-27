import numpy as np
import tensorflow as tf
from metrics import dice, jacobian_determinant
from eval.baseline_ants import run_baseline


# =========================
# EVALUATION VOXELMORPH
# =========================
def evaluate_voxelmorph(model, fixed_list, moving_list, n_samples=5):

    print("\n🚀 EVALUATION VOXELMORPH 3D")

    dice_scores = []
    jac_scores = []

    for i in range(min(n_samples, len(fixed_list))):

        fixed = tf.expand_dims(fixed_list[i], 0)
        moving = tf.expand_dims(moving_list[i], 0)

        warped, flow = model([moving, fixed], training=False)

        # Dice
        d = dice(fixed, warped).numpy()
        dice_scores.append(d)

        # Jacobian (folding risk)
        j = jacobian_determinant(flow).numpy()
        jac_scores.append(j)

        print(f"Sample {i} -> Dice: {d:.4f} | Folding: {j:.4f}")

    print("\n📊 VOXELMORPH SUMMARY:")
    print("Mean Dice:", np.mean(dice_scores))
    print("Mean Folding:", np.mean(jac_scores))

    return dice_scores, jac_scores


# =========================
# BASELINE ANTs COMPARISON
# =========================
def evaluate_ants(fixed_paths, moving_paths, n_samples=5):

    print("\n🧠 EVALUATION ANTs (SyN baseline)")

    scores = []

    for i in range(min(n_samples, len(fixed_paths))):

        warped, sim = run_baseline(
            fixed_paths[i],
            moving_paths[i]
        )

        scores.append(sim)
        print(f"Sample {i} -> ANTs similarity: {sim:.4f}")

    print("\n📊 ANTs SUMMARY:")
    print("Mean similarity:", np.mean(scores))

    return scores


# =========================
# FULL COMPARISON REPORT
# =========================
def full_evaluation(model, fixed, moving, fixed_paths=None, moving_paths=None):

    print("\n==============================")
    print("📌 FULL REGISTRATION EVALUATION")
    print("==============================")

    vox_scores = evaluate_voxelmorph(model, fixed, moving)

    ants_scores = None
    if fixed_paths is not None and moving_paths is not None:
        ants_scores = evaluate_ants(fixed_paths, moving_paths)

    print("\n==============================")
    print("🏁 FINAL REPORT")
    print("==============================")

    print("VoxelMorph Dice:", np.mean(vox_scores[0]))
    print("VoxelMorph Folding:", np.mean(vox_scores[1]))

    if ants_scores is not None:
        print("ANTs similarity:", np.mean(ants_scores))

    print("==============================")