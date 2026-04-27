import nibabel as nib
import numpy as np


def reorient_to_ras(path):
    img = nib.load(path)

    # force canonical orientation (TRÈS IMPORTANT)
    img = nib.as_closest_canonical(img)

    data = img.get_fdata().astype(np.float32)

    return data