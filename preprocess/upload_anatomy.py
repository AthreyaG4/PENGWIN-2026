import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "anatomy_segmentation"
HF_TOKEN = os.getenv("hf_auth_token")


def read_metadata(anatomy_dir: Path, split: str) -> dict[str, dict]:
    records = json.loads((anatomy_dir / split / "metadata.json").read_text())
    return {r["patient_id"]: r for r in records}


def make_train_generator(anatomy_dir: Path):
    def generator():
        split_dir = anatomy_dir / "train"
        meta_by_patient = read_metadata(anatomy_dir, "train")
        patch_files = sorted(split_dir.glob("patient_*_patch_*.npz"))

        print(
            f"Found {len(patch_files)} patch files for training in {anatomy_dir.name}."
        )

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

    return generator


def make_eval_generator(anatomy_dir: Path):
    def generator():
        split_dir = anatomy_dir / "validation"
        meta_by_patient = read_metadata(anatomy_dir, "validation")
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

    return generator


def upload(anatomy: str, hub_name: str) -> None:
    anatomy_dir = OUTPUT_DIR / anatomy

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                make_train_generator(anatomy_dir),
                writer_batch_size=100,
            ),
            "validation": Dataset.from_generator(
                make_eval_generator(anatomy_dir),
                writer_batch_size=5,
            ),
        }
    )
    dataset = dataset.with_format("numpy")

    print(f"\nUploading {anatomy} dataset to {hub_name} ...")
    dataset.push_to_hub(
        hub_name,
        private=False,
        num_shards={"train": 16, "validation": 2},
        token=HF_TOKEN,
    )
    print(f"Done: {hub_name}")


if __name__ == "__main__":
    upload("pelvic", "AG2307/pengwin-2026-anatomy-segmentation-pelvic")
    upload("femur", "AG2307/pengwin-2026-anatomy-segmentation-femur")
