import os
from pathlib import Path

from datasets import Dataset, Nifti
from dotenv import load_dotenv

load_dotenv()

NIFTI_DIR = Path(__file__).parent.parent / "dataset" / "nifti"
HF_TOKEN = os.getenv("hf_auth_token")
HF_REPO = "AG2307/pengwin-2026-nifti"


def collect_pairs() -> tuple[list[str], list[str]]:
    images_dir = NIFTI_DIR / "images"
    labels_dir = NIFTI_DIR / "labels"

    image_files = sorted(images_dir.glob("*_image.nii.gz"), key=lambda p: p.name)

    image_paths, label_paths = [], []
    for img_path in image_files:
        case_id = img_path.name.replace("_image.nii.gz", "")
        lbl_path = labels_dir / f"{case_id}_label.nii.gz"
        if not lbl_path.exists():
            print(f"[WARN] No label found for {case_id}, skipping.")
            continue
        image_paths.append(str(img_path))
        label_paths.append(str(lbl_path))

    return image_paths, label_paths


def main() -> None:
    image_paths, label_paths = collect_pairs()
    print(f"Found {len(image_paths)} image-label pairs.")

    ds = (
        Dataset.from_dict({"image": image_paths, "label": label_paths})
        .cast_column("image", Nifti())
        .cast_column("label", Nifti())
    )

    print(ds)
    print(f"\nUploading to {HF_REPO} ...")
    ds.push_to_hub(HF_REPO, private=False, token=HF_TOKEN, num_shards=20)
    print("Done.")


if __name__ == "__main__":
    main()
