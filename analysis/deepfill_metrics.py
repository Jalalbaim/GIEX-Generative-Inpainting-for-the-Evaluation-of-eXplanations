"""
deepfill_metrics.py

Evaluates DeepFill v2 inpainting quality per XAI attribution method.

For each method folder in latest_outputs_deepfill/{method}/deepfill/:
  - Matches each inpainted image to its GT counterpart in data/imagenette_sub15/
  - Runs VGG16 to get the GT class probability for both GT and deepfill images
  - Computes:
      AURC  — trapezoidal area under the retention curve (0% → K% deletion)
      AG@K  — Average Gain at K% deletion  (how much probability increased)
      AD@K  — Average Drop at K% deletion  (how much probability decreased)

Output: results/deepfill_metrics.csv

Usage:
    python deepfill_metrics.py [--pct_deleted 0.30]
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

DEEPFILL_ROOT    = "./latest_outputs_deepfill"
DATASET_ROOT     = "./data/imagenette_sub15"
VGG_PATH         = "data/weights/resnet50-11ad3fa6.pth"
CLASS_INDEX_PATH = "./data/weights/imagenet_class_index.json"
OUTPUT_CSV       = "./results/deepfill_metrics.csv"


def load_model(path: str):
    model = models.resnet50(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu"))
    model.eval()
    return model


class Evaluator:
    """Wraps ResNet50 to return softmax probability at a given class index."""

    def __init__(self, device: th.device, class_index_path: str):
        self.device = device
        with open(class_index_path) as f:
            idx_map = json.load(f)
        self.idx2label = [idx_map[str(i)][1] for i in range(len(idx_map))]
        self.model = load_model(VGG_PATH).to(device)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_logits(self, img_pil: Image.Image) -> th.Tensor:
        t = self.tf(img_pil).unsqueeze(0).to(self.device)
        with th.no_grad():
            return self.model(t).squeeze(0)

    def predict(self, logits: th.Tensor):
        idx = int(logits.argmax().item())
        return idx, self.idx2label[idx]

    def prob_at(self, logits: th.Tensor, idx: int) -> float:
        return float(F.softmax(logits, dim=0)[idx].item())


# ---------------------------------------------------------------------------
# Metric functions  (consistent with metrics_ag_ad_ai.py)
# ---------------------------------------------------------------------------

def metric_drop(val_gt: float, val_res: float) -> float:
    """Average Drop: normalised decrease in probability."""
    if val_gt == 0.0:
        return 0.0
    return max(0.0, val_gt - val_res) / val_gt


def metric_gain(val_gt: float, val_res: float) -> float:
    """Average Gain: normalised increase in probability."""
    den = 1.0 - val_gt
    return max(0.0, val_res - val_gt) / den if den > 0 else 0.0


def compute_aurc(prob_gt: float, prob_df: float, pct: float) -> float:
    """
    Two-point trapezoidal approximation of the retention curve area.
    x-axis: pixels removed (0 → pct*100 %)
    y-axis: classifier probability
    """
    x = np.array([0.0, pct * 100.0])
    y = np.array([prob_gt, prob_df])
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------

def build_gt_index(dataset_root: str) -> dict:
    """Return {stem: absolute_path} for all images in dataset_root."""
    index = {}
    for ext in ("*.JPEG", "*.jpg", "*.jpeg", "*.png"):
        for path in glob.glob(os.path.join(dataset_root, "**", ext),
                              recursive=True):
            stem = os.path.splitext(os.path.basename(path))[0]
            index[stem] = path
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pct_deleted: float):
    K = int(round(pct_deleted * 100))
    print(f"Deletion percentage : {K}%")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device             : {device}")

    evaluator = Evaluator(device, CLASS_INDEX_PATH)
    gt_index  = build_gt_index(DATASET_ROOT)
    print(f"GT images indexed  : {len(gt_index)}\n")

    methods = sorted([
        d for d in os.listdir(DEEPFILL_ROOT)
        if os.path.isdir(os.path.join(DEEPFILL_ROOT, d))
    ])
    print(f"Methods found: {methods}\n")

    records = []

    for method in methods:
        deepfill_dir = os.path.join(DEEPFILL_ROOT, method, "deepfill")
        if not os.path.isdir(deepfill_dir):
            print(f"[{method}] no deepfill/ folder — skipping")
            continue

        df_files = sorted(
            glob.glob(os.path.join(deepfill_dir, "*.png"))
            + glob.glob(os.path.join(deepfill_dir, "*.jpg"))
            + glob.glob(os.path.join(deepfill_dir, "*.jpeg"))
        )
        print(f"[{method}] {len(df_files)} images")

        aurcs, ags, ads = [], [], []

        for df_path in df_files:
            stem    = os.path.splitext(os.path.basename(df_path))[0]
            gt_path = gt_index.get(stem)
            if gt_path is None:
                print(f"  WARN: GT not found for {stem}")
                continue

            gt_pil = Image.open(gt_path).convert("RGB")
            df_pil = Image.open(df_path).convert("RGB")

            gt_logits       = evaluator.get_logits(gt_pil)
            gt_idx, gt_lbl  = evaluator.predict(gt_logits)
            prob_gt         = evaluator.prob_at(gt_logits, gt_idx)

            df_logits = evaluator.get_logits(df_pil)
            prob_df   = evaluator.prob_at(df_logits, gt_idx)

            aurcs.append(compute_aurc(prob_gt, prob_df, pct_deleted))
            ags.append(metric_gain(prob_gt, prob_df))
            ads.append(metric_drop(prob_gt, prob_df))

        n = len(aurcs)
        aurc_mean = float(np.mean(aurcs)) if n else float("nan")
        ag_mean   = float(np.mean(ags))   if n else float("nan")
        ad_mean   = float(np.mean(ads))   if n else float("nan")

        print(
            f"  n={n:3d}  AURC={aurc_mean:.4f}  "
            f"AG@{K}={ag_mean:.4f}  AD@{K}={ad_mean:.4f}"
        )

        records.append({
            "method":   method,
            "n_images": n,
            "AURC":     aurc_mean,
            f"AG@{K}":  ag_mean,
            f"AD@{K}":  ad_mean,
        })

    df_out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute AURC / AG@K / AD@K for DeepFill inpainting outputs."
    )
    parser.add_argument(
        "--pct_deleted",
        type=float,
        default=0.30,
        help="Fraction of pixels deleted when generating deepfill images (default: 0.30)",
    )
    args = parser.parse_args()
    main(args.pct_deleted)
