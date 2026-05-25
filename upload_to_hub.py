import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = Path(__file__).parent / "dataset" / "anatomy_segmentation"


def read_metadata(split: str) -> dict[str, dict]:
    records = json.loads((DATASET_DIR / split / "metadata.json").read_text())
    return {r["patient_id"]: r for r in records}


def train_generator():
    split_dir = DATASET_DIR / "train"
    meta_by_patient = read_metadata("train")
    patch_files = sorted(split_dir.glob("patient_*_patch_*.npz"))

    print(f"Found {len(patch_files)} patch files for training.")

    for f in patch_files:
        pid = f.stem.split("_patch_")[0]
        m = meta_by_patient[pid]
        npz = np.load(f)
        yield {
            "image": npz["image"],
            "label": npz["label"],
            "patient_id": pid,
            "original_spacing": m["original_spacing"],
            "original_shape": m["original_shape"],
            "original_affine": m["original_affine"],
        }


def eval_generator():
    split_dir = DATASET_DIR / "validation"
    meta_by_patient = read_metadata("validation")
    case_files = sorted(split_dir.glob("patient_???.npz"))

    for f in case_files:
        pid = f.stem
        m = meta_by_patient[pid]
        npz = np.load(f)
        yield {
            "image": npz["image"],
            "label": npz["label"],
            "patient_id": pid,
            "original_spacing": m["original_spacing"],
            "original_shape": m["original_shape"],
            "original_affine": m["original_affine"],
        }


dataset = DatasetDict(
    {
        "train": Dataset.from_generator(
            train_generator,
            writer_batch_size=100,
        ),
        "validation": Dataset.from_generator(
            eval_generator,
            writer_batch_size=5,
        ),
    }
)
dataset = dataset.with_format("numpy")

dataset.push_to_hub(
    "AG2307/pengwin-2026-anatomy-segmentation",
    private=False,
    num_shards={"train": 16, "validation": 4},
    token=os.getenv("hf_auth_token"),
)
