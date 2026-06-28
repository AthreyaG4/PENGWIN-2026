"""Generate and persist the train/validation split for all preprocessing pipelines.

Run this once before preprocess_anatomy.py or preprocess_fractures.py.
Writes splits.json next to this file, keyed by anatomy type and case directory name.
"""

import json
import random
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent / "19732767"
SPLITS_FILE = Path(__file__).parent / "splits.json"

SPLIT_RATIOS = {"train": 0.80, "validation": 0.20}
RANDOM_SEED = 42


def collect_cases(part_dirs: list[Path]) -> list[str]:
    cases = []
    for part_dir in part_dirs:
        cases.extend(
            sorted(
                (d.name for d in part_dir.iterdir() if d.is_dir()),
                key=lambda name: name,
            )
        )
    return cases


def make_split(cases: list[str]) -> dict[str, list[str]]:
    shuffled = cases.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled)
    train_n = round(len(shuffled) * SPLIT_RATIOS["train"])
    return {"train": shuffled[:train_n], "validation": shuffled[train_n:]}


def main() -> None:
    all_task1_parts = sorted(
        d
        for d in SOURCE_DIR.iterdir()
        if d.is_dir() and "task1" in d.name and "train_clicks" not in d.name
    )

    pelvic_parts = all_task1_parts[:2]
    femur_parts = all_task1_parts[2:]

    splits = {
        "pelvic": make_split(collect_cases(pelvic_parts)),
        "femur": make_split(collect_cases(femur_parts)),
    }

    SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    print(f"Wrote {SPLITS_FILE}")
    for anatomy, s in splits.items():
        print(
            f"  {anatomy}: {len(s['train'])} train, {len(s['validation'])} validation"
        )


if __name__ == "__main__":
    main()
