"""
Round-trip geometry test (MONAI only, no SimpleITK).

1. Read an MHA label file with MONAI.
2. Record original metadata (affine, spacing, shape).
3. Apply forward transforms (orient to RAS, resample to 1mm).
4. Create a fake prediction in transformed space.
5. Reverse transforms using only saved metadata.
6. Write output as .mha using MONAI's SaveImage.
7. Reload and assert geometry matches the original.

Usage:
    python test_roundtrip.py /path/to/case_dir
    (case_dir must contain image.mha and label.mha)
"""

import sys
from pathlib import Path
import time

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    SaveImage,
    Spacing,
)
from nibabel.orientations import aff2axcodes


def main(case_dir: str):
    case_dir = Path(case_dir)
    label_path = case_dir / "label.mha"
    assert label_path.exists(), f"Missing {label_path}"

    output_path = case_dir / "test_roundtrip_output.mha"

    # ------------------------------------------------------------------
    # Step 1: Load with MONAI and record metadata BEFORE any transforms
    # ------------------------------------------------------------------
    loader = LoadImage(image_only=True)
    ensure_ch = EnsureChannelFirst()

    label_raw = loader(str(label_path))
    label_raw = ensure_ch(label_raw)

    saved_affine = label_raw.affine.clone()
    saved_spacing = label_raw.pixdim.tolist()
    saved_shape = list(label_raw.shape[1:])  # spatial only
    saved_axcodes = "".join(aff2axcodes(saved_affine.numpy()))

    print("=== Saved metadata (pre-transform) ===")
    print(f"  Affine:\n{saved_affine.numpy()}")
    print(f"  Spacing:  {saved_spacing}")
    print(f"  Shape:    {saved_shape}")
    print(f"  Axcodes:  {saved_axcodes}")

    # ------------------------------------------------------------------
    # Step 2: Forward transforms (same as preprocessing pipeline)
    # ------------------------------------------------------------------
    orient_to_ras = Orientation(axcodes="RAS")
    resample_1mm = Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest")

    label_ras = orient_to_ras(label_raw)
    label_resampled = resample_1mm(label_ras)

    print("\n=== After forward transforms (RAS, 1mm) ===")
    print(f"  Affine:   \n{label_resampled.affine.numpy()}")
    print(f"  Shape:    {list(label_resampled.shape[1:])}")
    print(f"  Spacing:  {label_resampled.pixdim.tolist()}")
    print(f"  Axcodes:  {''.join(aff2axcodes(label_resampled.affine.numpy()))}")

    # ------------------------------------------------------------------
    # Step 3: Fake prediction in transformed space
    # ------------------------------------------------------------------
    spatial_shape = label_resampled.shape[1:]
    fake_pred = torch.randint(0, 5, (1, *spatial_shape), dtype=torch.float32)
    fake_pred = MetaTensor(fake_pred, affine=label_resampled.affine.clone())

    print(f"\n=== Fake prediction ===")
    print(f"  Shape: {list(fake_pred.shape[1:])}")

    # ------------------------------------------------------------------
    # Step 4: Reverse transforms using saved metadata
    # ------------------------------------------------------------------
    print(f"\n=== Reversing transforms ===")

    # 4a. Resample back to original spacing
    resample_back = Spacing(pixdim=saved_spacing, mode="nearest")
    pred_respaced = resample_back(fake_pred)
    print(f"  After resample  - shape: {list(pred_respaced.shape[1:])}")

    # 4b. Reorient back to original orientation
    orient_back = Orientation(axcodes=saved_axcodes)
    pred_reoriented = orient_back(pred_respaced)
    print(f"  After reorient  - shape: {list(pred_reoriented.shape[1:])}")
    print(
        f"  After reorient  - axcodes: "
        f"{''.join(aff2axcodes(pred_reoriented.affine.numpy()))}"
    )

    # 4c. Pad or crop to exact original shape (resampling can be off by a voxel)
    pred_np = pred_reoriented.squeeze(0).numpy()
    final = np.zeros(saved_shape, dtype=pred_np.dtype)
    slices = tuple(slice(0, min(o, r)) for o, r in zip(saved_shape, pred_np.shape))
    final[slices] = pred_np[slices]
    print(f"  After pad/crop  - shape: {list(final.shape)}")

    # ------------------------------------------------------------------
    # Step 5: Reconstruct MetaTensor with original affine and save
    # ------------------------------------------------------------------
    final_tensor = torch.from_numpy(final).unsqueeze(0)  # (1, D, H, W)
    final_meta = MetaTensor(final_tensor, affine=saved_affine)

    saver = SaveImage(
        output_postfix="",
        output_ext="",
        separate_folder=False,
        print_log=False,
    )
    saver(final_meta, filename=str(output_path))

    print(f"\n  Saved to {output_path}")

    # ------------------------------------------------------------------
    # Step 6: Reload and assert geometry matches original
    # ------------------------------------------------------------------
    time.sleep(3)  # Ensure file is fully written before reading
    reloaded = loader(str(output_path))
    reloaded = ensure_ch(reloaded)

    reloaded_affine = reloaded.affine.numpy()
    reloaded_spacing = reloaded.pixdim.tolist()
    reloaded_shape = list(reloaded.shape[1:])
    reloaded_axcodes = "".join(aff2axcodes(reloaded_affine))

    print("\n=== Assertions ===")

    # Affine
    np.testing.assert_allclose(
        reloaded_affine,
        saved_affine.numpy(),
        atol=1e-4,
        err_msg="Affine mismatch",
    )
    print("  Affine:    PASS")

    # Spacing
    for i, (a, b) in enumerate(zip(saved_spacing, reloaded_spacing)):
        assert abs(a - b) < 1e-4, f"Spacing mismatch axis {i}: original={a}, got={b}"
    print(f"  Spacing:   PASS  {reloaded_spacing}")

    # Shape
    assert reloaded_shape == saved_shape, (
        f"Shape mismatch: original={saved_shape}, got={reloaded_shape}"
    )
    print(f"  Shape:     PASS  {reloaded_shape}")

    # Orientation
    assert reloaded_axcodes == saved_axcodes, (
        f"Orientation mismatch: original={saved_axcodes}, got={reloaded_axcodes}"
    )
    print(f"  Axcodes:   PASS  {reloaded_axcodes}")

    print(f"\nAll assertions passed!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/case_dir")
        print("  case_dir must contain label.mha")
        sys.exit(1)
    main(sys.argv[1])
