"""
halluc.py — Visualise inpainting hallucinations at multiple mask percentages
for a single image, using both RePaint (guided diffusion) and DeepFill v2.

Saves to hallucination/
  original_256.png          — the input image at 256×256
  mask_<pct>.png            — binary mask (white=keep)
  masked_<pct>.png          — image with masked region zeroed out
  repaint_<pct>.png         — RePaint inpainted result
  deepfill_<pct>.png        — DeepFill v2 inpainted result

Usage:
    python halluc.py
    python halluc.py --conf_path confs/conf_sal_0.yml \\
                     --image data/10306/n01440764_ILSVRC2012_val_00010306.JPEG \\
                     --method saliency --out_dir hallucination
"""

import os
import sys
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)
from modules.Evaluation_Method import SaliencyAttributor
from knockout import KnockoutMaskGenerator

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
VGG_PATH      = "data/weights/vgg16-397923af.pth"
DEEPFILL_CKPT = "deepfillv2-pytorch/pretrained/states_tf_places2.pth"
IMG_SIZE      = 256
MEANS         = [0.485, 0.456, 0.406]
STDS          = [0.229, 0.224, 0.225]
PCTS          = list(range(10, 100, 10))   # 10, 20, …, 90


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def load_vgg(path: str):
    model = models.vgg16(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu"))
    model.eval()
    return model


def to_uint8(t: th.Tensor) -> np.ndarray:
    """Diffusion [-1,1] batch tensor → uint8 numpy [H,W,3]."""
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def pil_to_diffusion_tensor(pil_img: Image.Image, device: th.device) -> th.Tensor:
    """PIL RGB 256×256 → [1,3,256,256] float32 in [-1,1]."""
    t = T.ToTensor()(pil_img).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def pil_to_norm_tensor(pil_img: Image.Image) -> th.Tensor:
    """PIL RGB 256×256 → [1,3,256,256] ImageNet-normalised float32."""
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEANS, std=STDS),
    ])
    return tf(pil_img).unsqueeze(0)


def apply_mask_to_image(img_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
    """Zero out masked pixels.  mask_np: float [H,W], 1=keep 0=removed."""
    out = img_np.copy()
    out[mask_np == 0] = 0
    return Image.fromarray(out)


# --------------------------------------------------------------------------- #
# DeepFill v2                                                                  #
# --------------------------------------------------------------------------- #

def load_deepfill(ckpt_path: str, device: th.device):
    """Load the DeepFill v2 (TF-style) generator from a checkpoint."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "deepfillv2-pytorch"))
    ckpt = th.load(ckpt_path, map_location="cpu")
    g_state = ckpt["G"]

    if "stage1.conv1.conv.weight" in g_state:
        from model.networks import Generator
    else:
        from model.networks_tf import Generator

    gen = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    gen.load_state_dict(g_state, strict=True)
    gen.eval()
    return gen


def deepfill_inpaint(
    generator, img_pil: Image.Image, mask_np: np.ndarray, device: th.device
) -> Image.Image:
    """
    Run DeepFill v2 inpainting.

    mask_np : float [H,W]  1=keep  0=masked   (project convention)
    DeepFill expects       1=hole  0=valid     → invert
    """
    image = T.ToTensor()(img_pil).unsqueeze(0)          # [1,3,H,W] ∈ [0,1]

    # invert: hole=1, valid=0
    df_mask = th.from_numpy(1.0 - mask_np).float()      # [H,W]
    df_mask = df_mask.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]

    h, w = image.shape[2], image.shape[3]
    grid = 8
    image   = image[:, :3, :h//grid*grid, :w//grid*grid].to(device)
    df_mask = df_mask[:, :,  :h//grid*grid, :w//grid*grid].to(device)

    image   = image * 2.0 - 1.0                          # → [-1,1]
    df_mask = (df_mask > 0.5).float()

    image_masked = image * (1.0 - df_mask)
    ones_x = th.ones_like(image_masked)[:, 0:1]
    x_in   = th.cat([image_masked, ones_x, ones_x * df_mask], dim=1)  # 5-ch input

    with th.inference_mode():
        _, x_stage2 = generator(x_in, df_mask)

    image_out = image * (1.0 - df_mask) + x_stage2 * df_mask
    img_out   = ((image_out[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
    img_out   = img_out.to(dtype=th.uint8).cpu().numpy()
    return Image.fromarray(img_out)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Hallucination visualisation script")
    parser.add_argument("--conf_path", type=str, default="confs/conf_sal_0.yml",
                        help="Path to RePaint YAML config")
    parser.add_argument("--image", type=str,
                        default="data/10306/n01440764_ILSVRC2012_val_00010306.JPEG",
                        help="Input image path")
    parser.add_argument("--method", type=str, default="saliency",
                        help="XAI attribution method for mask generation")
    parser.add_argument("--out_dir", type=str, default="hallucination",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load and resize the input image                                      #
    # ------------------------------------------------------------------ #
    img_pil_256 = Image.open(args.image).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    img_np = np.array(img_pil_256)               # uint8 [256,256,3]
    img_pil_256.save(os.path.join(args.out_dir, "original_256.png"))
    print(f"Loaded image: {args.image}  ({img_np.shape})")

    # Diffusion tensor (kept on CPU until needed per iteration)
    gt_cpu = pil_to_diffusion_tensor(img_pil_256, th.device("cpu"))  # [1,3,256,256]

    # ------------------------------------------------------------------ #
    # Compute saliency attribution once (reused across all percentages)   #
    # ------------------------------------------------------------------ #
    vgg     = load_vgg(VGG_PATH)
    inp_norm = pil_to_norm_tensor(img_pil_256)   # [1,3,256,256] normalised
    attr    = SaliencyAttributor(model=vgg, method=args.method).compute(inp_norm)
    print(f"Attribution computed  (method={args.method}, shape={attr.shape})")

    # ------------------------------------------------------------------ #
    # Load RePaint diffusion model                                         #
    # ------------------------------------------------------------------ #
    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(args.conf_path))

    repaint_model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    repaint_model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    repaint_model.to(device)
    if conf.use_fp16:
        repaint_model.convert_to_fp16()
    repaint_model.eval()
    print("RePaint diffusion model loaded")

    # Optional guidance classifier
    cond_fn = None
    if conf.classifier_scale and conf.classifier_scale > 0 and conf.classifier_path:
        clf = create_classifier(**select_args(conf, classifier_defaults().keys()))
        clf.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        clf.to(device)
        if conf.classifier_use_fp16:
            clf.convert_to_fp16()
        clf.eval()

        def cond_fn(x, t, y=None, **_):
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = clf(x_in, t)
                logp   = F.log_softmax(logits, dim=-1)
                sel    = logp[range(len(logits)), y.view(-1)]
                return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale

        print("Guidance classifier loaded")

    def model_fn(x, t, y=None, gt=None, **_):
        return repaint_model(x, t, y if conf.class_cond else None, gt=gt)

    sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop

    # ------------------------------------------------------------------ #
    # Load DeepFill v2 generator                                           #
    # ------------------------------------------------------------------ #
    deepfill_gen = load_deepfill(DEEPFILL_CKPT, device)
    print("DeepFill v2 generator loaded")

    # ------------------------------------------------------------------ #
    # Iterate over mask percentages                                        #
    # ------------------------------------------------------------------ #
    gt = gt_cpu.to(device)

    for pct in PCTS:
        k = int(IMG_SIZE * IMG_SIZE * pct / 100)
        masker = KnockoutMaskGenerator(K=k, means=MEANS, stds=STDS)
        _, mask_t, _ = masker.generate(inp_norm, attr)   # mask_t: tensor [H,W]
        mask_np = mask_t.numpy().astype(np.float32)       # float [256,256]

        # --- Save mask and masked image --------------------------------- #
        mask_vis = (mask_np * 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(
            os.path.join(args.out_dir, f"mask_{pct:02d}.png")
        )

        masked_pil = apply_mask_to_image(img_np, mask_np)
        masked_pil.save(os.path.join(args.out_dir, f"masked_{pct:02d}.png"))

        # --- RePaint inpainting ---------------------------------------- #
        mask_rp = th.from_numpy(mask_np).float().to(device)
        mask_rp = mask_rp.unsqueeze(0).unsqueeze(0)       # [1,1,256,256]

        y_classes = (
            th.ones(1, dtype=th.long, device=device) * conf.cond_y
            if conf.cond_y is not None
            else th.randint(0, NUM_CLASSES, (1,), device=device)
        )

        result = sample_fn(
            model_fn,
            (1, 3, IMG_SIZE, IMG_SIZE),
            clip_denoised=conf.clip_denoised,
            model_kwargs={"gt": gt, "gt_keep_mask": mask_rp, "y": y_classes},
            cond_fn=cond_fn,
            device=device,
            progress=conf.show_progress,
            return_all=True,
            conf=conf,
        )
        repaint_np  = to_uint8(result["sample"])[0]       # [256,256,3]
        repaint_pil = Image.fromarray(repaint_np)
        repaint_pil.save(os.path.join(args.out_dir, f"repaint_{pct:02d}.png"))

        # --- DeepFill v2 inpainting ------------------------------------ #
        df_pil = deepfill_inpaint(deepfill_gen, img_pil_256, mask_np, device)
        df_pil.save(os.path.join(args.out_dir, f"deepfill_{pct:02d}.png"))

        print(
            f"[{pct:2d}%]  mask  ✓  masked  ✓  repaint  ✓  deepfill  ✓  "
            f"→ {args.out_dir}/"
        )

    print(f"\nDone. All results saved in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
