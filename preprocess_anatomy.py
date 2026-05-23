import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)

SOURCE_DIR = Path(__file__).parent / "19732767"
OUTPUT_DIR = Path(__file__).parent / "dataset" / "anatomy_segmentation"

SPLITS = ["train", "validation", "test"]
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
RANDOM_SEED = 42

HU_MIN, HU_MAX = -175, 250
TARGET_SPACING = (1.0, 1.0, 1.0)
PATCH_SIZE = (96, 96, 96)
PATCHES_PER_VOLUME = 4
POS_RATIO = 2
NEG_RATIO = 1


class RemapLabelsd(MapTransform):
    """Collapses instance-segmentation IDs into semantic class IDs.

    Instance encoding:  1–50  → sacrum (1)
                        51–100 → left hip (2)
                       101–150 → right hip (3)
                       151–200 → femur (4)
                             0 → background (0)
    """

    _RANGES = [(1, 50, 1), (51, 100, 2), (101, 150, 3), (151, 200, 4)]

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            out = torch.zeros_like(lbl)
            for lo, hi, semantic in self._RANGES:
                out[(lbl >= lo) & (lbl <= hi)] = semantic
            d[key] = (
                MetaTensor(out, meta=lbl.meta) if isinstance(lbl, MetaTensor) else out
            )
        return d


_load = LoadImaged(keys=["image", "label"])
_ensure_channel = EnsureChannelFirstd(keys=["image", "label"])
_remap = RemapLabelsd(keys=["label"])
_orient = Orientationd(keys=["image", "label"], axcodes="RAS")
_spacing = Spacingd(
    keys=["image", "label"],
    pixdim=TARGET_SPACING,
    mode=("bilinear", "nearest"),
)
_scale = ScaleIntensityRanged(
    keys=["image"],
    a_min=HU_MIN,
    a_max=HU_MAX,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)
_crop_fg = CropForegroundd(keys=["image", "label"], source_key="image")
_rand_crop = RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=PATCH_SIZE,
    pos=POS_RATIO,
    neg=NEG_RATIO,
    num_samples=PATCHES_PER_VOLUME,
)


def preprocess(image_src: Path, label_src: Path) -> tuple[dict, dict]:
    data = {"image": str(image_src), "label": str(label_src)}
    data = _load(data)
    data = _ensure_channel(data)
    data = _remap(data)
    meta = {
        "original_spacing": data["image"].pixdim.tolist(),
        "original_shape": list(data["image"].shape),
        "original_affine": data["image"].affine.tolist(),
    }
    data = _orient(data)
    data = _spacing(data)
    data = _crop_fg(data)
    data = _scale(data)
    return data, meta


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


def convert_case_train(case_dir: Path, patient_id: int) -> dict | None:
    image_src = case_dir / "image.mha"
    label_src = case_dir / "label.mha"

    if not image_src.exists() or not label_src.exists():
        print(f"[WARN] Missing files in {case_dir}, skipping.")
        return None

    split_dir = OUTPUT_DIR / "train"
    data, meta = preprocess(image_src, label_src)
    patches = _rand_crop(data)

    for k, patch in enumerate(patches):
        np.savez_compressed(
            split_dir / f"patient_{patient_id:03d}_patch_{k:02d}.npz",
            image=patch["image"].numpy().astype(np.float32),
            label=patch["label"].numpy().astype(np.uint8),
        )

    return {"patient_id": f"patient_{patient_id:03d}", **meta}


def convert_case_eval(case_dir: Path, split: str, patient_id: int) -> dict | None:
    image_src = case_dir / "image.mha"
    label_src = case_dir / "label.mha"

    if not image_src.exists() or not label_src.exists():
        print(f"[WARN] Missing files in {case_dir}, skipping.")
        return None

    split_dir = OUTPUT_DIR / split
    data, meta = preprocess(image_src, label_src)

    np.savez_compressed(
        split_dir / f"patient_{patient_id:03d}.npz",
        image=data["image"].numpy().astype(np.float32),
        label=data["label"].numpy().astype(np.uint8),
    )

    return {"patient_id": f"patient_{patient_id:03d}", **meta}


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
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    patient_id = 1

    for split in SPLITS:
        combined = pelvic_splits[split] + femur_splits[split]
        total = len(combined)
        split_metadata = []

        print(f"\n--- {split.upper()} ({total} cases) ---")
        for i, case_dir in enumerate(combined, 1):
            fracture_type = "pelvic" if i <= len(pelvic_splits[split]) else "femur"
            if split == "train":
                meta = convert_case_train(case_dir, patient_id)
            else:
                meta = convert_case_eval(case_dir, split, patient_id)
            if meta is not None:
                split_metadata.append(meta)
                print(
                    f"  [{i:03d}/{total}] {fracture_type} | "
                    f"{case_dir.parent.name}/{case_dir.name} → patient_{patient_id:03d}"
                )
                patient_id += 1

        (OUTPUT_DIR / split / "metadata.json").write_text(
            json.dumps(split_metadata, indent=2)
        )

    print(f"\nDone. {patient_id - 1} patients written to {OUTPUT_DIR}/")
    for split in SPLITS:
        n_pelvic = len(pelvic_splits[split])
        n_femur = len(femur_splits[split])
        print(f"  {split}: {n_pelvic + n_femur} ({n_pelvic} pelvic + {n_femur} femur)")


if __name__ == "__main__":
    main()
