from pathlib import Path

import SimpleITK as sitk

SOURCE_DIR = Path(__file__).parent.parent / "19732767"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "nifti"

PELVIC_PARTS = [
    SOURCE_DIR / "PENGWIN26_task1_2_train_part1",
    SOURCE_DIR / "PENGWIN26_task1_2_train_part2",
]
FEMUR_PARTS = [
    SOURCE_DIR / "PENGWIN26_task1_2_train_part3",
    SOURCE_DIR / "PENGWIN26_task1_2_train_part4",
]


def collect_cases(part_dirs: list[Path]) -> list[Path]:
    cases = []
    for part_dir in part_dirs:
        cases.extend(
            sorted((d for d in part_dir.iterdir() if d.is_dir()), key=lambda p: p.name)
        )
    return cases


def convert_case(case_dir: Path, images_dir: Path, labels_dir: Path) -> None:
    image_src = case_dir / "image.mha"
    label_src = case_dir / "label.mha"

    if not image_src.exists() or not label_src.exists():
        print(f"[WARN] Missing files in {case_dir}, skipping.")
        return

    case_id = case_dir.name

    image = sitk.ReadImage(str(image_src))
    sitk.WriteImage(image, str(images_dir / f"{case_id}_image.nii.gz"))

    label = sitk.ReadImage(str(label_src))
    sitk.WriteImage(label, str(labels_dir / f"{case_id}_label.nii.gz"))

    print(f"  Saved {case_id}")


def main() -> None:
    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    pelvic_cases = collect_cases(PELVIC_PARTS)
    femur_cases = collect_cases(FEMUR_PARTS)

    print(f"Pelvic cases: {len(pelvic_cases)}")
    print(f"Femur cases:  {len(femur_cases)}")

    print("\n--- Pelvic ---")
    for case_dir in pelvic_cases:
        convert_case(case_dir, images_dir, labels_dir)

    print("\n--- Femur ---")
    for case_dir in femur_cases:
        convert_case(case_dir, images_dir, labels_dir)

    print(f"\nDone. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
