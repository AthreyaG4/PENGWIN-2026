import json
import os
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "fracture_segmentation"
HF_TOKEN = os.getenv("hf_auth_token")

_CASE_ID_RE = re.compile(r"^(patient_\d{3})_(.+?)(?:_patch_\d{2})?$")


def _parse_stem(stem: str) -> tuple[str, str]:
    m = _CASE_ID_RE.match(stem)
    return m.group(1), m.group(2)


def read_metadata(output_dir: Path, split: str) -> dict[str, dict]:
    records = json.loads((output_dir / split / "metadata.json").read_text())
    return {r["case_id"]: r for r in records}


def make_generator(output_dir: Path, split: str):
    def generator():
        split_dir = output_dir / split
        meta_by_case = read_metadata(output_dir, split)
        case_files = sorted(split_dir.glob("patient_*.npz"))

        for f in case_files:
            case_id, bone_name = _parse_stem(f.stem)
            m = meta_by_case[case_id]
            bone_meta = next((b for b in m["bones"] if b["bone"] == bone_name), {})
            npz = np.load(f)
            yield {
                "image": npz["image"],
                "gt_label": npz["gt_label"],
                "fdm_weights": npz["fdm_weights"],
                "case_id": case_id,
                "bone": bone_name,
                "num_fragments": bone_meta.get("num_fragments", -1),
                "original_spacing": m["original_spacing"],
                "original_shape": m["original_shape"],
                "original_affine": m["original_affine"],
            }

    return generator


def upload(hub_name: str) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                make_generator(OUTPUT_DIR, "train"),
                writer_batch_size=100,
            ),
            "validation": Dataset.from_generator(
                make_generator(OUTPUT_DIR, "validation"),
                writer_batch_size=5,
            ),
        }
    )
    dataset = dataset.with_format("numpy")

    print(f"\nUploading dataset to {hub_name} ...")
    dataset.push_to_hub(
        hub_name,
        private=False,
        num_shards={"train": 4, "validation": 2},
        token=HF_TOKEN,
    )
    print(f"Done: {hub_name}")


if __name__ == "__main__":
    upload("AG2307/pengwin-2026-fracture-segmentation")
