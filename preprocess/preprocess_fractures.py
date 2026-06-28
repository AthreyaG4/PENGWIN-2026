import json
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from scipy.ndimage import distance_transform_edt
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

SOURCE_DIR = Path(__file__).parent.parent / "19732767"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "fracture_segmentation"

SPLITS = ["train", "validation"]
SPLITS_FILE = Path(__file__).parent / "splits.json"

HU_MIN, HU_MAX = -200, 1400
TARGET_SPACING = (0.83, 0.83, 0.89)
PATCH_SIZE = (96, 96, 96)
LAMBDA_BACK = 0.2
LAMBDA_FDM = 10.0
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
_pad = SpatialPadd(
    keys=["image", "gt_label", "fdm_weights"], spatial_size=PATCH_SIZE, mode="constant"
)
_rand_crop_contact = RandCropByPosNegLabeld(
    keys=["image", "gt_label", "fdm_weights"],
    label_key="contact_map",
    spatial_size=PATCH_SIZE,
    pos=POS_RATIO,
    neg=NEG_RATIO,
    num_samples=PATCHES_CONTACT,
)
_rand_crop_fg = RandCropByPosNegLabeld(
    keys=["image", "gt_label", "fdm_weights"],
    label_key="gt_label",
    spatial_size=PATCH_SIZE,
    pos=POS_RATIO,
    neg=NEG_RATIO,
    num_samples=PATCHES_NO_CONTACT,
)


def load_volume(image_src: Path, label_src: Path) -> tuple[dict, dict]:
    data = {"image": str(image_src), "label": str(label_src)}
    data = _load(data)
    data = _ensure_channel(data)
    meta = {
        "original_spacing": data["image"].pixdim.tolist(),
        "original_shape": list(data["image"].shape[1:]),
        "original_affine": data["image"].affine.tolist(),
    }
    data = _orient(data)
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


def collapse_to_major_minor(instance_label: torch.Tensor) -> torch.Tensor:
    """0 = background, 1 = major fragment (largest), 2 = minor fragments."""
    arr = instance_label.numpy()
    ids, counts = np.unique(arr[arr > 0], return_counts=True)
    if len(ids) == 0:
        return torch.zeros_like(instance_label)

    major_id = ids[np.argmax(counts)]
    gt = np.zeros_like(arr, dtype=np.uint8)
    gt[arr == major_id] = 1
    gt[(arr > 0) & (arr != major_id)] = 2
    return torch.as_tensor(gt, dtype=instance_label.dtype)


def find_cfs(instance_label: np.ndarray) -> np.ndarray:
    """Boolean mask: True where a foreground voxel borders a voxel of a different fragment.

    Must be called with instance labels (1..N), not collapsed major/minor labels,
    so that minor-minor fragment contacts are also detected.
    """
    arr = np.squeeze(instance_label)  # (D, H, W)
    foreground = arr > 0
    cfs = np.zeros_like(arr, dtype=bool)
    for axis in range(3):
        for shift in (1, -1):
            neighbor = np.roll(arr, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            neighbor[tuple(edge)] = 0
            cfs |= foreground & (neighbor > 0) & (neighbor != arr)
    return cfs.reshape(instance_label.shape)


def compute_fdm(gt_label: np.ndarray, cfs: np.ndarray) -> np.ndarray:
    """Euclidean distance from every foreground voxel to nearest CFS, normalized to [0, 1].

    CFS voxels have distance 0; the deepest interior voxel has distance 1.
    Background stays 0.
    """
    foreground = np.squeeze(gt_label) > 0
    cfs_3d = np.squeeze(cfs).astype(bool)
    dist = distance_transform_edt(~cfs_3d, sampling=TARGET_SPACING)  # distance in mm
    fdm = np.where(foreground, dist, 0.0).astype(np.float32)
    max_dist = fdm.max()
    if max_dist > 0:
        fdm /= max_dist
    return fdm.reshape(gt_label.shape)


def fdm_to_weights(
    gt_label: np.ndarray,
    fdm: np.ndarray,
    lambda_back: float = LAMBDA_BACK,
    lambda_fdm: float = LAMBDA_FDM,
) -> np.ndarray:
    """Per-voxel loss weights from the FDM paper formula.

    CFS voxels (fdm=0) → weight ≈ 1.0; deep interior (fdm=1) → ≈ λ_back; background → λ_back.
    Normalized so the sum of foreground weights equals the foreground voxel count.
    """
    foreground = (np.squeeze(gt_label) >= 1).astype(np.float32)
    fdm_3d = np.squeeze(fdm).astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(lambda_fdm * fdm_3d - 5.0))
    W = lambda_back + foreground * (1.0 - lambda_back) * sigmoid
    fg_mask = foreground > 0
    fg_sum = W[fg_mask].sum()
    fg_count = foreground.sum()
    if fg_sum > 0:
        W[fg_mask] = W[fg_mask] * fg_count / fg_sum
    return W.reshape(gt_label.shape).astype(np.float32)


def normalize_and_save(
    image: torch.Tensor,
    gt_label: torch.Tensor,
    fdm_weights: torch.Tensor,
    out_stem: Path,
) -> None:
    """Pad to ≥ PATCH_SIZE, then crop to exactly PATCH_SIZE if needed."""
    meta = image.meta if isinstance(image, MetaTensor) else {}

    def as_meta(t: torch.Tensor) -> torch.Tensor:
        return (
            MetaTensor(t.float(), meta=meta)
            if isinstance(image, MetaTensor)
            else t.float()
        )

    padded = _pad(
        {
            "image": image,
            "gt_label": as_meta(gt_label),
            "fdm_weights": as_meta(fdm_weights),
        }
    )
    img_p, gt_p, wt_p = padded["image"], padded["gt_label"], padded["fdm_weights"]

    if tuple(img_p.shape[1:]) == PATCH_SIZE:
        np.savez_compressed(
            str(out_stem) + ".npz",
            image=img_p.numpy().astype(np.float32),
            gt_label=gt_p.numpy().astype(np.uint8),
            fdm_weights=wt_p.numpy().astype(np.float32),
        )
        return

    has_minor = (gt_p == 2).any().item()

    if has_minor:
        minor_map = (gt_p == 2).float()
        if isinstance(gt_p, MetaTensor):
            minor_map = MetaTensor(minor_map, meta=gt_p.meta)
        patches = _rand_crop_contact(
            {
                "image": img_p,
                "gt_label": gt_p,
                "fdm_weights": wt_p,
                "contact_map": minor_map,
            }
        )
    else:
        patches = _rand_crop_fg({"image": img_p, "gt_label": gt_p, "fdm_weights": wt_p})

    for k, patch in enumerate(patches):
        np.savez_compressed(
            str(out_stem) + f"_patch_{k:02d}.npz",
            image=patch["image"].numpy().astype(np.float32),
            gt_label=patch["gt_label"].numpy().astype(np.uint8),
            fdm_weights=patch["fdm_weights"].numpy().astype(np.float32),
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
    splits_data = json.loads(SPLITS_FILE.read_text())
    # Map (anatomy_key → case_name) → (Path, bones) for fast lookup.
    # anatomy_key matches the splits.json keys: "pelvic" or "femur".
    anatomy_map: dict[str, dict[str, tuple[Path, list]]] = {}
    for case_dir, bones in cases:
        bone_names = tuple(b[0] for b in bones)
        anatomy_key = "femur" if bone_names == ("femur",) else "pelvic"
        anatomy_map.setdefault(anatomy_key, {})[case_dir.name] = (case_dir, bones)

    train, validation = [], []
    for anatomy_key, by_name in anatomy_map.items():
        for split, names in splits_data[anatomy_key].items():
            entries = [by_name[n] for n in names if n in by_name]
            if split == "train":
                train.extend(entries)
            else:
                validation.extend(entries)

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
        instance_arr = bone_data["label"].numpy()
        gt_label = collapse_to_major_minor(bone_data["label"])

        gt_arr = gt_label.numpy()
        cfs = find_cfs(instance_arr)
        fdm = compute_fdm(gt_arr, cfs)
        fdm_weights = torch.as_tensor(fdm_to_weights(gt_arr, fdm))

        if split == "train":
            normalize_and_save(
                bone_data["image"],
                gt_label,
                fdm_weights,
                split_dir / f"{case_id}_{bone_name}",
            )
        else:
            np.savez_compressed(
                split_dir / f"{case_id}_{bone_name}.npz",
                image=bone_data["image"].numpy().astype(np.float32),
                gt_label=gt_arr.astype(np.uint8),
                fdm_weights=fdm_weights.numpy().astype(np.float32),
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
