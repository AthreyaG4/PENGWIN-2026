from pathlib import Path
import numpy as np

BASE = Path("Experiment 2/dataset/fracture_segmentation")

for split in ("train", "validation"):
    files = sorted((BASE / split).glob("patient_*.npz"))
    has_minor = sum(1 for f in files if np.load(f)["gt_label"].max() >= 2)
    print(f"[{split}]")
    print(f"  Total files : {len(files)}")
    print(f"  Has minor   : {has_minor}")
    print(f"  No minor    : {len(files) - has_minor}")
    print()
