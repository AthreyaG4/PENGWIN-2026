import json
import random
from pathlib import Path

import SimpleITK as sitk

SOURCE_DIR = Path(__file__).parent / "19732767"
OUTPUT_DIR = Path(__file__).parent / "dataset"

SPLITS = ["train", "validation", "test"]
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}

RANDOM_SEED = 42

HU_MIN, HU_MAX = -1000, 400
TARGET_SPACING = (1.0, 1.0, 1.0)


def resample(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    interpolator,
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        round(original_size[i] * original_spacing[i] / target_spacing[i])
        for i in range(3)
    ]

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def collect_cases(part_dirs: list[Path]) -> list[Path]:
    cases = []
    for part_dir in part_dirs:
        cases.extend(
            sorted((d for d in part_dir.iterdir() if d.is_dir()), key=lambda p: p.name)
        )
    return cases


def split_cases(cases: list[Path]) -> dict[str, list[Path]]:
    shuffled = cases.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled)
    n = len(shuffled)
    train_n = round(n * SPLIT_RATIOS["train"])
    val_n = round(n * SPLIT_RATIOS["validation"])
    return {
        "train": shuffled[:train_n],
        "validation": shuffled[train_n : train_n + val_n],
        "test": shuffled[train_n + val_n :],
    }


def convert_case(
    case_dir: Path,
    split: str,
    patient_id: int,
    meta_f,
) -> bool:
    image_src = case_dir / "image.mha"
    label_src = case_dir / "label.mha"

    if not image_src.exists() or not label_src.exists():
        print(f"[WARN] Missing files in {case_dir}, skipping.")
        return False

    split_dir = OUTPUT_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    img_name = f"patient_{patient_id:03d}_ct.nii.gz"
    lbl_name = f"patient_{patient_id:03d}_label.nii.gz"

    img = sitk.ReadImage(str(image_src), sitk.sitkFloat32)
    img = sitk.Clamp(img, sitk.sitkFloat32, HU_MIN, HU_MAX)
    img = resample(img, TARGET_SPACING, sitk.sitkLinear)
    sitk.WriteImage(img, str(images_dir / img_name))

    lbl = sitk.ReadImage(str(label_src), sitk.sitkUInt8)
    lbl = resample(lbl, TARGET_SPACING, sitk.sitkNearestNeighbor)
    sitk.WriteImage(lbl, str(labels_dir / lbl_name))

    row = {
        "file_name": f"images/{img_name}",
        "label_file_name": f"labels/{lbl_name}",
    }
    meta_f.write(json.dumps(row) + "\n")
    meta_f.flush()
    return True


def main():
    all_task1_parts = sorted(
        d
        for d in SOURCE_DIR.iterdir()
        if d.is_dir() and "task1" in d.name and "train_clicks" not in d.name
    )

    pelvic_parts = all_task1_parts[:2]
    femur_parts = all_task1_parts[2:]

    pelvic_cases = collect_cases(pelvic_parts)
    femur_cases = collect_cases(femur_parts)

    pelvic_splits = split_cases(pelvic_cases)
    femur_splits = split_cases(femur_cases)

    for split in SPLITS:
        split_dir = OUTPUT_DIR / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    patient_id = 1

    for split in SPLITS:
        meta_path = OUTPUT_DIR / split / "metadata.jsonl"
        combined = pelvic_splits[split] + femur_splits[split]
        total = len(combined)

        print(f"\n--- {split.upper()} ({total} cases) ---")
        with meta_path.open("w") as meta_f:
            for i, case_dir in enumerate(combined, 1):
                fracture_type = "pelvic" if i <= len(pelvic_splits[split]) else "femur"
                ok = convert_case(case_dir, split, patient_id, meta_f)
                if ok:
                    print(
                        f"  [{i:03d}/{total}] {fracture_type} | "
                        f"{case_dir.parent.name}/{case_dir.name} → patient_{patient_id:03d}"
                    )
                    patient_id += 1

    print(f"\nDone. {patient_id - 1} patients written to {OUTPUT_DIR}/")
    for split in SPLITS:
        n_pelvic = len(pelvic_splits[split])
        n_femur = len(femur_splits[split])
        print(f"  {split}: {n_pelvic + n_femur} ({n_pelvic} pelvic + {n_femur} femur)")


if __name__ == "__main__":
    main()
