from pathlib import Path

from datasets import Dataset, DatasetDict, Nifti

DATASET_DIR = Path(__file__).parent / "dataset"


def load_split(split: str) -> Dataset:
    split_dir = DATASET_DIR / split
    image_files = sorted(split_dir.glob("images/*.nii.gz"))
    label_files = sorted(split_dir.glob("labels/*.nii.gz"))

    ds = Dataset.from_dict(
        {
            "nifti": [str(f) for f in image_files],
            "label": [str(f) for f in label_files],
        }
    )
    ds = ds.cast_column("nifti", Nifti())
    ds = ds.cast_column("label", Nifti())
    return ds


dataset = DatasetDict(
    {
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    }
)

dataset.push_to_hub(
    "AG2307/pengwin-2026-nifti",
    private=False,
    num_shards={"train": 16, "validation": 2, "test": 2},
)
