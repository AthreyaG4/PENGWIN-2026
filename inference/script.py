from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from monai.data import MetaTensor
from monai.data.image_writer import ITKWriter
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet, SwinUNETR
from monai.transforms import (
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    ResampleToMatch,
    ScaleIntensityRange,
    Spacing,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("/opt/ml/model")
INPUT_PATH = Path("/input/images/peripelvic-fracture-ct")
OUTPUT_PATH = Path("/output/images/peripelvic-fracture-ct-segmentation")

HU_MIN, HU_MAX = -200, 1400
TARGET_SPACING = (0.83, 0.83, 0.89)
PATCH_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
OVERLAP = 0.5

PELVIC_NUM_LABELS = 4
FEMUR_NUM_LABELS = 2
FDM_NUM_LABELS = 3

# ~1 cm³ at TARGET_SPACING: 1000 mm³ / (0.83 × 0.83 × 0.89 mm³) ≈ 1631 voxels
VOXEL_THRESHOLD = 1639

# Label offset per anatomy class ID — final label = offset + instance_id
# (major fragment → offset+1, minor fragments → offset+2, offset+3, …)
ANATOMY_OFFSETS: dict[str, dict[int, int]] = {
    "p": {1: 0, 2: 50, 3: 100},  # sacrum, left hip, right hip
    "f": {1: 150},  # femur
}


def load_model(name: str, num_labels: int) -> SwinUNETR:
    model = SwinUNETR(
        in_channels=1,
        out_channels=num_labels,
        feature_size=48,
        use_v2=True,
        use_checkpoint=False,
    )
    checkpoint = torch.load(MODEL_DIR / name, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_fdm_model() -> DynUNet:
    kernel_size = [[3, 3, 3]] * 5
    strides = [[1, 1, 1]] + [[2, 2, 2]] * 4
    upsample_kernel_size = [[2, 2, 2]] * 4
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=FDM_NUM_LABELS,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        deep_supervision=True,
        deep_supr_num=3,
        res_block=True,
    )
    checkpoint = torch.load(
        MODEL_DIR / "best_model_fdm.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(input_path: Path) -> tuple[MetaTensor, MetaTensor]:
    """
    Returns:
        preprocessed: MetaTensor (1, H, W, D) in resampled RAS space — fed to models.
        original_ref:  MetaTensor (1, H, W, D) in native patient space — used as
                       the reference grid when restoring the output to original spacing.
    """
    load = LoadImage(image_only=True)
    ensure_channel = EnsureChannelFirst()
    orient = Orientation(axcodes="RAS")
    spacing = Spacing(pixdim=TARGET_SPACING, mode="bilinear")
    scale = ScaleIntensityRange(
        a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True
    )

    image = load(str(input_path))
    image = ensure_channel(image)
    original_ref = image.clone()  # snapshot BEFORE any spatial transforms

    image = orient(image)
    image = spacing(image)
    image = scale(image)
    return image, original_ref


def classify_scan(original_ref: MetaTensor) -> str:
    """
    Rule-based pelvic/femur classification from PENGWIN 2026 competition guidelines.
    Must be called with the pre-resampling image so spacing and dims are original.
    """
    pixdim = original_ref.pixdim
    spacing_x, spacing_y, spacing_z = (
        pixdim[0].item(),
        pixdim[1].item(),
        pixdim[2].item(),
    )
    _, dim_x, _, dim_z = original_ref.shape
    physical_x_mm = spacing_x * dim_x
    physical_z_mm = spacing_z * dim_z

    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "p"
        elif spacing_z <= 0.90:
            return "f"
        else:
            return "p" if spacing_y <= 0.91 else "f"
    else:
        if spacing_z <= 0.68:
            return "p" if physical_z_mm <= 193.55 else "f"
        else:
            return "p" if physical_z_mm <= 390.78 else "f"


@torch.no_grad()
def run_inference(image_tensor: MetaTensor, model: SwinUNETR) -> np.ndarray:
    """Returns argmax segmentation (H, W, D) as uint8."""
    tensor = image_tensor.unsqueeze(0).to(device)  # (1, 1, H, W, D)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        pred = sliding_window_inference(
            inputs=tensor,
            roi_size=PATCH_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian",
        )
    return pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def get_bounding_box(mask: np.ndarray, pad: int = 5) -> tuple[slice, ...] | None:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad, np.array(mask.shape) - 1)
    return tuple(slice(mn, mx + 1) for mn, mx in zip(mins, maxs))


def extract_anatomy_crops(
    image_np: np.ndarray, pred: np.ndarray, mode: str
) -> dict[int, dict]:
    """
    Extract per-anatomy image crops from the Stage 1 prediction.

    Returns dict keyed by class ID:
        {"name": str, "image": np.ndarray (H,W,D), "mask": np.ndarray, "bbox": tuple[slice]}
    """
    class_ids = (
        {1: "Sacrum", 2: "Left Hip", 3: "Right Hip"} if mode == "p" else {1: "Femur"}
    )
    crops = {}
    for cid, name in class_ids.items():
        mask = (pred == cid).astype(np.uint8)
        bbox = get_bounding_box(mask)
        if bbox is None:
            continue
        cropped_image = image_np[0][bbox].copy()
        cropped_mask = mask[bbox].copy()
        crops[cid] = {
            "name": name,
            "image": cropped_image * cropped_mask,  # zero outside anatomy mask
            "mask": cropped_mask,
            "bbox": bbox,
        }
    return crops


@torch.no_grad()
def run_fdm_inference(crop_image: np.ndarray, model: DynUNet) -> np.ndarray:
    """
    crop_image: (H, W, D) masked CT crop, already normalised.
    Returns segmentation (H, W, D): 0=background, 1=major fragment, 2=minor fragments.
    """
    tensor = (
        torch.from_numpy(crop_image).float().unsqueeze(0).unsqueeze(0).to(device)
    )  # (1, 1, H, W, D)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        pred = sliding_window_inference(
            inputs=tensor,
            roi_size=PATCH_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian",
        )
        # DynUNet with deep_supervision returns (1, num_heads, C, H, W, D) in train mode;
        # in eval mode it returns (1, C, H, W, D). Guard handles either case.
        if pred.dim() == 6:
            pred = pred[:, 0]
    return pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def run_all_fdm(crops: dict[int, dict], fdm_model: DynUNet) -> dict[int, np.ndarray]:
    return {
        cid: run_fdm_inference(crop["image"], fdm_model) for cid, crop in crops.items()
    }


def assign_instance_labels(pred: np.ndarray, offset: int) -> np.ndarray:
    """
    Convert a per-anatomy FDM prediction (0=bg, 1=major, 2=minor) into
    PENGWIN instance labels using the anatomy's label offset.

    Major fragment  → offset + 1  (single label, no CCA needed)
    Minor fragments → offset + 2, offset + 3, … (one label per CC,
                      components smaller than VOXEL_THRESHOLD are dropped)
    """
    instance_map = np.zeros_like(pred, dtype=np.uint8)
    instance_map[pred == 1] = offset + 1

    minor_mask = pred == 2
    if not minor_mask.any():
        return instance_map

    # 26-connectivity so diagonal contacts within one fragment stay merged
    components, num_components = scipy_label(minor_mask, structure=np.ones((3, 3, 3)))

    # Sort surviving components by descending volume so the largest minor
    # fragment always gets the lowest label within this anatomy's range.
    minor_fragments = [
        (i, int((components == i).sum()))
        for i in range(1, num_components + 1)
        if (components == i).sum() >= VOXEL_THRESHOLD
    ]
    minor_fragments.sort(key=lambda x: x[1], reverse=True)

    for new_label, (old_id, _) in enumerate(minor_fragments, start=offset + 2):
        instance_map[components == old_id] = new_label

    return instance_map


def stitch_predictions(
    full_shape: tuple[int, int, int],
    crops: dict[int, dict],
    fdm_preds: dict[int, np.ndarray],
    mode: str,
) -> np.ndarray:
    offsets = ANATOMY_OFFSETS[mode]
    output = np.zeros(full_shape, dtype=np.uint8)
    for cid, pred in fdm_preds.items():
        output[crops[cid]["bbox"]] = assign_instance_labels(pred, offsets[cid])
    return output


def save_output(
    stitched_np: np.ndarray,
    preprocessed_image: MetaTensor,
    original_ref: MetaTensor,
    output_path: Path,
) -> None:
    """
    1. Wraps the stitched uint8 prediction in the resampled-space affine.
    2. ResampleToMatch undoes both Spacing and Orientation — restores native
       patient spacing, origin, and direction.
    3. ITKWriter writes the result as MHA, preserving the correct ITK metadata.
    """
    stitched_meta = MetaTensor(
        torch.from_numpy(stitched_np).unsqueeze(0).float(),
        affine=preprocessed_image.meta["affine"],
    )
    output_native = ResampleToMatch(mode="nearest")(stitched_meta, original_ref)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ITKWriter(output_dtype=np.uint8)
    writer.set_data_array(output_native, channel_dim=0)
    writer.set_metadata(output_native.meta, resample=False)
    writer.write(str(output_path))


def main() -> None:
    # Locate the single .mha file placed by Grand Challenge in the input directory
    input_files = sorted(INPUT_PATH.glob("*.mha"))
    if not input_files:
        raise FileNotFoundError(f"No .mha file found in {INPUT_PATH}")
    input_file = input_files[0]
    output_file = OUTPUT_PATH / input_file.name

    # 1. Preprocess — capture original ref for final resampling
    image_tensor, original_ref = preprocess_image(input_file)

    # 2. Classify scan type using original spacing/dims (before resampling)
    mode = classify_scan(original_ref)

    # 3. Stage 1: semantic anatomy segmentation
    ckpt = "best_model_pelvic.pth" if mode == "p" else "best_model_femur.pth"
    num_labels = PELVIC_NUM_LABELS if mode == "p" else FEMUR_NUM_LABELS
    stage1_model = load_model(ckpt, num_labels)
    stage1_pred = run_inference(image_tensor, stage1_model)  # (H, W, D) uint8
    del stage1_model
    torch.cuda.empty_cache()

    # 7. Restore native spacing/affine and save (Stage 1 pred only for now)
    save_output(stage1_pred, image_tensor, original_ref, output_file)

    # # 4. Extract per-anatomy crops from Stage 1 prediction
    # image_np = image_tensor.numpy()  # (1, H, W, D)
    # crops = extract_anatomy_crops(image_np, stage1_pred, mode)

    # # 5. Stage 2: major/minor fragment segmentation per anatomy
    # fdm_model = load_fdm_model()
    # fdm_preds = run_all_fdm(crops, fdm_model)
    # del fdm_model
    # torch.cuda.empty_cache()

    # # 6. Stitch per-anatomy predictions into full-volume label
    # full_shape = tuple(image_tensor.shape[1:])  # (H, W, D) in resampled space
    # stitched = stitch_predictions(full_shape, crops, fdm_preds, mode)

    # # 7. Restore native spacing/affine and save
    # save_output(stitched, image_tensor, original_ref, output_file)


if __name__ == "__main__":
    main()
