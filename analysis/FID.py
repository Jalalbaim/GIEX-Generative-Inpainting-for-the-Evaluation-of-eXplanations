"""
FID.py

For each of the 7 XAI methods:
  1. Pick 10 random images from data/imagenette_sub15
  2. Generate GT / masked / blurred / repainted / deepfill variants
  3. Compute FID(GT, variant) for each of the 4 variants

Output
------
  fid_outputs/{method}/{GT,masked,blurred,repainted,deepfill}/  (images)
  fid_results.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import glob
import json
import random
import sys
import warnings

import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms, models
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from scipy import linalg

# project imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.Evaluation_Method import SaliencyAttributor
from modules.knockout import KnockoutMaskGenerator
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)
import conf_mgt
from utils import yamlread

# constants
DATASET_ROOT   = "./data/imagenette_sub15"
OUTPUT_ROOT    = "./fid_outputs"
CONF_PATH      = "./confs/conf_sal_0.yml"   # diffusion settings (method-agnostic)
VGG_PATH       = "data/weights/vgg16-397923af.pth"
DEEPFILL_PATH  = "deepfillv2-pytorch/pretrained/states_pt_places2.pth"
INCEPTION_PATH = "data/weights/inception_v3_google-0cc3c7bd.pth"

N_IMAGES    = 10
PCT         = 0.30
BLUR_RADIUS = 10
SEED        = 42
IMAGE_SIZE  = 256

METHODS  = ["gbp", "grad-cam", "grads", "integrated gradients",
            "saliency", "smoothgrad", "lrp"]
VARIANTS = ["repainted", "deepfill", "masked", "blurred"]
FOLDERS  = ["GT", "masked", "blurred", "repainted", "deepfill"]

MEANS = [0.485, 0.456, 0.406]
STDS  = [0.229, 0.224, 0.225]

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

OUTPUT_CSV = "fid_results.csv"

FID_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ImageFolder(Dataset):
    def __init__(self, folder: str, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class InceptionFeatureExtractor(th.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=None)
        state_dict = th.load(INCEPTION_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        self.blocks = th.nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            th.nn.MaxPool2d(kernel_size=3, stride=2),
            model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            th.nn.MaxPool2d(kernel_size=3, stride=2),
            model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c,
            model.Mixed_6d, model.Mixed_6e,
            model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
        )
        self.pool = th.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)   # (B, 2048)


@th.no_grad()
def compute_statistics(folder: str, model: th.nn.Module):
    dataset = ImageFolder(folder, FID_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True)
    all_feats = []
    for batch in loader:
        batch = batch.to(DEVICE)
        feats = model(batch).cpu().numpy()
        all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    mu    = np.mean(all_feats, axis=0)
    sigma = np.cov(all_feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn("FID: sqrtm produced non-finite values; adding epsilon to diagonal.")
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def load_vgg(path: str):
    model = models.vgg16(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu", weights_only=False))
    return model.eval()


def load_deepfill(checkpoint_path: str, device):
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfillv2-pytorch")
    if _repo not in sys.path:
        sys.path.append(_repo)

    raw   = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw["G"] if "G" in raw else raw

    if "stage1.conv1.conv.weight" in state:
        from model.networks import Generator
        generator = Generator(cnum_in=5, cnum_out=3, cnum=48, return_flow=False)
    else:
        from model.networks_tf import Generator
        generator = Generator(cnum_in=5, cnum=48, return_flow=False)

    generator.load_state_dict(state)
    return generator.to(device).eval()


def deepfill_inpaint(generator, img_np: np.ndarray, mask_np: np.ndarray, device) -> np.ndarray:
    H, W = img_np.shape[:2]
    H8, W8 = H // 8 * 8, W // 8 * 8

    img_t   = (th.from_numpy(img_np[:H8, :W8]).float().permute(2, 0, 1)
                .unsqueeze(0) / 127.5 - 1.0).to(device)
    df_mask = (th.from_numpy((1.0 - mask_np[:H8, :W8]).astype(np.float32))
                .unsqueeze(0).unsqueeze(0).to(device))

    img_masked = img_t * (1.0 - df_mask)
    ones = th.ones_like(df_mask)
    x    = th.cat([img_masked, ones, ones * df_mask], dim=1)

    with th.inference_mode():
        _, x_stage2 = generator(x, df_mask)

    result_t    = img_t * (1.0 - df_mask) + x_stage2 * df_mask
    result_crop = ((result_t.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0)
                   * 127.5).clip(0, 255).astype(np.uint8)

    if H8 == H and W8 == W:
        return result_crop
    out = img_np.copy()
    out[:H8, :W8] = result_crop
    return out


def to_uint8(t):
    """Diffusion tensor (B,C,H,W) in [-1,1] → numpy (H,W,3) uint8."""
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def make_output_dirs(method_root: str):
    for folder in FOLDERS:
        os.makedirs(os.path.join(method_root, folder), exist_ok=True)


def build_variants(img_np: np.ndarray, method_xai: str, vgg_model, k: int):
    """Return (masked_np, blurred_np, mask_np, mask_tensor)."""
    img_pil = Image.fromarray(img_np)
    prep = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS, std=STDS),
    ])
    inp = prep(img_pil).unsqueeze(0)

    attributor = SaliencyAttributor(model=vgg_model, method=method_xai)
    attr       = attributor.compute(inp)

    masker = KnockoutMaskGenerator(K=k, means=MEANS, stds=STDS)
    out_knock, mask, _ = masker.generate(inp, attr)

    masked_np = (out_knock.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

    mask_np      = mask.numpy()
    blurred_full = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)))
    blurred_np   = np.where(mask_np[:, :, None] == 0, blurred_full, img_np).astype(np.uint8)

    return masked_np, blurred_np, mask_np, mask


def generate_images(
    image_paths,
    method_xai: str,
    method_root: str,
    vgg_model,
    diffusion_model,
    diffusion,
    cond_fn,
    deepfill_model,
    conf,
):
    """Generate and save all 5 image types for each input image."""
    make_output_dirs(method_root)
    k = int(IMAGE_SIZE * IMAGE_SIZE * PCT)
    print(f"  Deletion rate: {int(PCT * 100)}%  → K={k} pixels")

    prep_diff = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  Processing: {stem}")

        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_np  = np.array(img_pil)

        # GT
        Image.fromarray(img_np).save(os.path.join(method_root, "GT", f"{stem}.png"))

        # masked & blurred
        masked_np, blurred_np, mask_np, mask = build_variants(img_np, method_xai, vgg_model, k)
        Image.fromarray(masked_np).save(os.path.join(method_root, "masked",  f"{stem}.png"))
        Image.fromarray(blurred_np).save(os.path.join(method_root, "blurred", f"{stem}.png"))

        # repainted (diffusion)
        gt_t    = (th.from_numpy(img_np).float().permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(DEVICE)
        mask_4d = th.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

        y_classes = (
            th.ones(1, dtype=th.long, device=DEVICE) * conf.cond_y
            if conf.cond_y is not None
            else th.randint(0, NUM_CLASSES, (1,), device=DEVICE)
        )

        sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop

        def model_fn(x, t, y=None, gt=None, **_):
            return diffusion_model(x, t, y if conf.class_cond else None, gt=gt)

        result = sample_fn(
            model_fn,
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
            clip_denoised=conf.clip_denoised,
            model_kwargs={"gt": gt_t, "gt_keep_mask": mask_4d, "y": y_classes},
            cond_fn=cond_fn,
            device=DEVICE,
            progress=False,
            return_all=True,
            conf=conf,
        )
        repainted_np = to_uint8(result["sample"])[0]
        Image.fromarray(repainted_np).save(os.path.join(method_root, "repainted", f"{stem}.png"))

        # deepfill
        deepfill_np = deepfill_inpaint(deepfill_model, img_np, mask_np, DEVICE)
        Image.fromarray(deepfill_np).save(os.path.join(method_root, "deepfill", f"{stem}.png"))

        print(f"    Saved GT / masked / blurred / repainted / deepfill → {method_root}/")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(SEED)

    print(f"Device: {DEVICE}")

    # pick 10 random images 
    all_images = sorted(set(
        glob.glob(os.path.join(DATASET_ROOT, "**", "*.JPEG"), recursive=True)
        + glob.glob(os.path.join(DATASET_ROOT, "**", "*.jpg"),  recursive=True)
        + glob.glob(os.path.join(DATASET_ROOT, "**", "*.jpeg"), recursive=True)
        + glob.glob(os.path.join(DATASET_ROOT, "**", "*.png"),  recursive=True)
    ))
    if len(all_images) == 0:
        raise FileNotFoundError(f"No images found under {DATASET_ROOT}")
    selected = random.sample(all_images, min(N_IMAGES, len(all_images)))
    print(f"Selected {len(selected)} images from {DATASET_ROOT}")
    for p in selected:
        print(f"  {os.path.basename(p)}")

    # load config (diffusion settings) 
    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(CONF_PATH))

    # load diffusion model (once) 
    print("\nLoading diffusion model …")
    diff_model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    diff_model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    diff_model.to(DEVICE)
    if conf.use_fp16:
        diff_model.convert_to_fp16()
    diff_model.eval()
    print("Diffusion model loaded.")

    # load guidance classifier (once)
    cond_fn = None
    if conf.classifier_scale > 0 and conf.classifier_path:
        classifier = create_classifier(**select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        classifier.to(DEVICE)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, **_):
            with th.enable_grad():
                x_in  = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                logp   = F.log_softmax(logits, dim=-1)
                sel    = logp[range(len(logits)), y.view(-1)]
                return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale

        print("Guidance classifier loaded.")

    # load VGG (once, shared across methods)
    print("Loading VGG16 …")
    vgg_model = load_vgg(VGG_PATH)
    print("VGG16 loaded.")

    # load DeepFill (once) 
    print("Loading DeepFill v2 …")
    deepfill_model = load_deepfill(DEEPFILL_PATH, DEVICE)
    print("DeepFill v2 loaded.")

    # load Inception extractor for FID 
    print("Loading Inception v3 for FID …")
    inception = InceptionFeatureExtractor().to(DEVICE).eval()
    print("Inception v3 loaded.\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    rows    = []
    results = {}

    # per-method loop
    for method in METHODS:
        print(f"\n══ Method: {method} ══")
        method_root = os.path.join(OUTPUT_ROOT, method)

        # generate images
        generate_images(
            image_paths=selected,
            method_xai=method,
            method_root=method_root,
            vgg_model=vgg_model,
            diffusion_model=diff_model,
            diffusion=diffusion,
            cond_fn=cond_fn,
            deepfill_model=deepfill_model,
            conf=conf,
        )

        # compute FID
        gt_dir = os.path.join(method_root, "GT")
        print(f"  Computing GT statistics …")
        mu_gt, sigma_gt = compute_statistics(gt_dir, inception)

        results[method] = {}

        for variant in VARIANTS:
            variant_dir = os.path.join(method_root, variant)
            print(f"  Computing FID(GT, {variant}) …")
            try:
                mu_v, sigma_v = compute_statistics(variant_dir, inception)
                fid = frechet_distance(mu_gt, sigma_gt, mu_v, sigma_v)
                results[method][variant] = round(fid, 4)
                rows.append({"method": method, "variant": variant, "FID": round(fid, 4)})
                print(f"    FID = {fid:.4f}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                results[method][variant] = None
                rows.append({"method": method, "variant": variant, "FID": None})

    # save CSV 
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "variant", "FID"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {OUTPUT_CSV}")

    with open("fid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("JSON saved → fid_results.json")

    # print summary table
    print("\ FID Summary")
    header = f"{'Method':<25}" + "".join(f"{v:>15}" for v in VARIANTS)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        row = f"{method:<25}"
        for v in VARIANTS:
            val = results.get(method, {}).get(v)
            row += f"{str(round(val, 2)) if val is not None else 'N/A':>15}"
        print(row)


if __name__ == "__main__":
    main()
