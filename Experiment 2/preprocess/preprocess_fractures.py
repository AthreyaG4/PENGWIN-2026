import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import (
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)

SOURCE_DIR = Path(__file__).parent.parent.parent / "19732767"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "fracture_segmentation"

SPLITS = ["train", "validation"]
SPLIT_RATIOS = {"train": 0.80, "validation": 0.20}
RANDOM_SEED = 42

HU_MIN, HU_MAX = -200, 1400
TARGET_SPACING = (1.0, 1.0, 1.0)
CSM_KERNEL_SIZE = 7
PATCH_SIZE = (96, 96, 96)
PATCHES_CONTACT = 2
PATCHES_NO_CONTACT = 1
POS_RATIO = 1
NEG_RATIO = 1

PELVIC_BONES = [
    ("sacrum", 1, 50),
    ("left_hip", 51, 100),
    ("right_hip", 101, 150),
]
FEMUR_BONES = [
    ("femur", 151, 200),
]

_load = LoadImaged(keys=["image", "label"])
_ensure_channel = EnsureChannelFirstd(keys=["image", "label"])
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
_crop_fg = CropForegroundd(keys=["image", "label"], source_key="label")
_pad = SpatialPadd(keys=["image", "csm"], spatial_size=PATCH_SIZE, mode="constant")
_rand_crop_contact = RandCropByPosNegLabeld(
    keys=["image", "csm"],
    label_key="contact_map",
    spatial_size=PATCH_SIZE,
    pos=POS_RATIO,
    neg=NEG_RATIO,
    num_samples=PATCHES_CONTACT,
)
_rand_crop_fg = RandCropByPosNegLabeld(
    keys=["image", "csm"],
    label_key="csm",
    spatial_size=PATCH_SIZE,
    pos=POS_RATIO,
    neg=NEG_RATIO,
    num_samples=PATCHES_NO_CONTACT,
)


def load_volume(image_src: Path, label_src: Path) -> tuple[dict, dict]:
    data = {"image": str(image_src), "label": str(label_src)}
    data = _load(data)
    data = _ensure_channel(data)
    data = _orient(data)
    meta = {
        "original_spacing": data["image"].pixdim.tolist(),
        "original_shape": list(data["image"].shape),
        "original_affine": data["image"].affine.tolist(),
    }
    data = _spacing(data)
    data = _scale(data)
    return data, meta


def extract_bone(data: dict, lo: int, hi: int) -> dict | None:
    """Mask the CT and relabel fragments 1..N for a single bone's label range."""
    lbl = data["label"]
    img = data["image"]
    bone_mask = (lbl >= lo) & (lbl <= hi)

    if not bone_mask.any():
        return None

    new_label = torch.zeros_like(lbl)
    unique_vals = lbl[bone_mask].unique().sort().values
    for new_id, old_val in enumerate(unique_vals.tolist(), start=1):
        new_label[lbl == int(old_val)] = new_id

    masked_img = img.clone()
    masked_img[~bone_mask.expand_as(masked_img)] = 0.0

    bone_data = {
        "image": masked_img,
        "label": MetaTensor(new_label, meta=lbl.meta)
        if isinstance(lbl, MetaTensor)
        else new_label,
    }
    return _crop_fg(bone_data)


def compute_csm(
    instance_label: torch.Tensor, kernel_size: int = CSM_KERNEL_SIZE
) -> torch.Tensor:
    """0 = background, 1 = foreground, 2 = contact surface.

    A voxel is contact if it is foreground and any neighbour within `kernel_size`
    belongs to a different non-zero instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half_k = kernel_size // 2

    t = instance_label.float().to(device)  # (1, D, H, W)
    t5 = t.unsqueeze(0)  # (1, 1, D, H, W)
    padded = F.pad(t5, (half_k,) * 6, value=0)

    D, H, W = t.shape[1:]
    contact = torch.zeros_like(t, dtype=torch.bool)

    for dx in range(-half_k, half_k + 1):
        for dy in range(-half_k, half_k + 1):
            for dz in range(-half_k, half_k + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                shifted = padded[
                    :,
                    :,
                    half_k + dx : half_k + dx + D,
                    half_k + dy : half_k + dy + H,
                    half_k + dz : half_k + dz + W,
                ].squeeze(0)
                contact |= (t > 0) & (shifted > 0) & (shifted != t)

    csm = torch.zeros_like(instance_label)
    csm[instance_label > 0] = 1
    csm[contact.to(instance_label.device)] = 2
    return csm


def normalize_and_save(
    image: torch.Tensor,
    csm: torch.Tensor,
    out_stem: Path,
) -> None:
    """Pad to ≥ PATCH_SIZE, then crop to exactly PATCH_SIZE if needed."""
    meta = image.meta if isinstance(image, MetaTensor) else {}
    csm_t = (
        MetaTensor(csm.float(), meta=meta)
        if isinstance(image, MetaTensor)
        else csm.float()
    )

    padded = _pad({"image": image, "csm": csm_t})
    img_p, csm_p = padded["image"], padded["csm"]

    if tuple(img_p.shape[1:]) == PATCH_SIZE:
        np.savez_compressed(
            str(out_stem) + ".npz",
            image=img_p.numpy().astype(np.float32),
            csm=csm_p.numpy().astype(np.uint8),
        )
        return

    has_contact = (csm_p == 2).any().item()

    if has_contact:
        contact_map = (csm_p == 2).float()
        if isinstance(csm_p, MetaTensor):
            contact_map = MetaTensor(contact_map, meta=csm_p.meta)
        patches = _rand_crop_contact(
            {"image": img_p, "csm": csm_p, "contact_map": contact_map}
        )
    else:
        patches = _rand_crop_fg({"image": img_p, "csm": csm_p})

    for k, patch in enumerate(patches):
        np.savez_compressed(
            str(out_stem) + f"_patch_{k:02d}.npz",
            image=patch["image"].numpy().astype(np.float32),
            csm=patch["csm"].numpy().astype(np.uint8),
        )


def collect_cases(
    part_dirs: list[Path], bones: list[tuple[str, int, int]]
) -> list[tuple[Path, list[tuple[str, int, int]]]]:
    cases = []
    for part_dir in part_dirs:
        for d in sorted(
            (d for d in part_dir.iterdir() if d.is_dir()), key=lambda p: p.name
        ):
            cases.append((d, bones))
    return cases


def split_cases(
    cases: list[tuple[Path, list]],
) -> dict[str, list[tuple[Path, list]]]:
    # Stratify by anatomy so each split has equal pelvic and femur counts.
    # Group key is the tuple of bone names, e.g. ("sacrum","left_hip","right_hip") or ("femur",).
    groups: dict[tuple, list] = {}
    for case in cases:
        key = tuple(b[0] for b in case[1])
        groups.setdefault(key, []).append(case)

    train, validation = [], []
    for group in groups.values():
        shuffled = group.copy()
        random.seed(RANDOM_SEED)
        random.shuffle(shuffled)
        n = len(shuffled)
        train_n = round(n * SPLIT_RATIOS["train"])
        train.extend(shuffled[:train_n])
        validation.extend(shuffled[train_n:])

    return {"train": train, "validation": validation}


def process_case(
    case_dir: Path,
    bones: list[tuple[str, int, int]],
    split_dir: Path,
    split: str,
    patient_id: int,
) -> dict | None:
    image_src = case_dir / "image.mha"
    label_src = case_dir / "label.mha"

    if not image_src.exists() or not label_src.exists():
        print(f"[WARN] Missing files in {case_dir}, skipping.")
        return None

    case_id = f"patient_{patient_id:03d}"
    data, meta = load_volume(image_src, label_src)

    bone_records = []
    for bone_name, lo, hi in bones:
        bone_data = extract_bone(data, lo, hi)
        if bone_data is None:
            continue

        num_fragments = int(bone_data["label"].max().item())
        csm = compute_csm(bone_data["label"])

        if split == "train":
            normalize_and_save(
                bone_data["image"],
                csm,
                split_dir / f"{case_id}_{bone_name}",
            )
        else:
            np.savez_compressed(
                split_dir / f"{case_id}_{bone_name}.npz",
                image=bone_data["image"].numpy().astype(np.float32),
                csm=csm.numpy().astype(np.uint8),
            )

        bone_records.append({"bone": bone_name, "num_fragments": num_fragments})

    if not bone_records:
        return None

    return {"case_id": case_id, "bones": bone_records, **meta}


def process_all(cases: list[tuple[Path, list]], output_dir: Path) -> None:
    splits = split_cases(cases)

    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    patient_id = 1

    for split in SPLITS:
        split_cases_list = splits[split]
        total = len(split_cases_list)
        split_metadata = []
        split_dir = output_dir / split

        print(f"\n--- {split.upper()} ({total} cases) ---")
        for i, (case_dir, bones) in enumerate(split_cases_list, 1):
            meta = process_case(case_dir, bones, split_dir, split, patient_id)
            if meta is not None:
                split_metadata.append(meta)
                bone_names = [b["bone"] for b in meta["bones"]]
                print(
                    f"  [{i:03d}/{total}] {case_dir.parent.name}/{case_dir.name}"
                    f" → patient_{patient_id:03d} {bone_names}"
                )
                patient_id += 1

        (output_dir / split / "metadata.json").write_text(
            json.dumps(split_metadata, indent=2)
        )

    print(f"\nDone. {patient_id - 1} patients written to {output_dir}/")
    for split in SPLITS:
        print(f"  {split:12s}: {len(splits[split]):3d}")


def main():
    all_task1_parts = sorted(
        d
        for d in SOURCE_DIR.iterdir()
        if d.is_dir() and "task1" in d.name and "train_clicks" not in d.name
    )

    pelvic_parts = all_task1_parts[:2]
    femur_parts = all_task1_parts[2:]

    all_cases = collect_cases(pelvic_parts, PELVIC_BONES) + collect_cases(
        femur_parts, FEMUR_BONES
    )

    process_all(all_cases, OUTPUT_DIR)


if __name__ == "__main__":
    main()
