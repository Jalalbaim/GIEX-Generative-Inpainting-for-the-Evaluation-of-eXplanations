"""
RePaint AURC
@author:J.BAIM
"""

import os
import argparse
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.colors import ListedColormap
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
from modules.knockout import KnockoutMaskGenerator
import skimage as ski

CLASS_INDEX_PATH = "./data/weights/imagenet_class_index.json"
path = "data/weights/resnet50-11ad3fa6.pth"

def load_model(path):
    model = models.resnet50(weights=None)
    state_dict = th.load(path, map_location=th.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def to_uint8(t):
    if t is None:
        return None
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def mask_to_uint8(t):
    return (t.squeeze().cpu().numpy() * 255).astype(np.uint8)


def imagenet_denorm(t):
    means = th.tensor([0.485, 0.456, 0.406], device=t.device).view(1,3,1,1)
    stds  = th.tensor([0.229, 0.224, 0.225], device=t.device).view(1,3,1,1)
    return t * stds + means


def generate_knockout(img_np, xai, k):
    img_pil = Image.fromarray(img_np)
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    prep = transforms.Compose([
        transforms.Resize(img_np.shape[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ])
    inp = prep(img_pil).unsqueeze(0)
    heatmap = SaliencyAttributor(
        model=load_model(path),
        method=xai
    )
    attr = heatmap.compute(inp)
    # masque + knockout
    masker = KnockoutMaskGenerator(K=k, means=means, stds=stds)
    out_knock, mask, _ = masker.generate(inp, attr)
    return out_knock, mask

def L2_img (img1, img2):
    """
    Compute the L2 distance between two images.
    img1, img2: numpy arrays of shape (H, W, C)
    Returns the L2 distance as a float.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    return np.sqrt(np.sum((img1 - img2) ** 2))

class LogitEvaluator:

    def __init__(self, device: th.device, class_index_path: str):
        self.device = device
        with open(class_index_path, 'r') as f:
            idx_map = json.load(f)
        self.idx2label = [idx_map[str(i)][1] for i in range(len(idx_map))]

        self.model = load_model(path)
        self.model.to(self.device)

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def evaluate(self, img_pil: Image.Image) -> np.ndarray:

        t = self.tf(img_pil).unsqueeze(0).to(self.device)
        with th.no_grad():
            logits = self.model(t)
        return logits.cpu().numpy().flatten()

    def predict(self, logits: np.ndarray):

        idx = int(np.argmax(logits))
        return idx, self.idx2label[idx]

    @staticmethod
    def softmax_score(logits: np.ndarray, idx: int) -> float:

        probs = F.softmax(th.from_numpy(logits), dim=0)
        return float(probs[idx].item())



def main(conf: conf_mgt.Default_Conf):
    print("Start", conf["name"] )
    device = dist_util.dev(conf.get("device"))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("Diffusion model loaded")

    classifier = None
    if conf.classifier_scale > 0 and conf.classifier_path:
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys())
        )
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        print("Classifier loaded")

    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset="eval", dsName=eval_name)
    print("Dataloader size:", len(dl))

    for batch in dl:
        for k,v in batch.items():
            if isinstance(v, th.Tensor):
                batch[k] = v.to(device)
        gt = batch["GT"]
        img_name = batch["GT_name"][0].split('.')[0]

        img_np = to_uint8(gt)[0]

        # paramètres XAI
        method_xai = conf.get("method_xai", "saliency")
        print("XAI method:", method_xai)

        if conf.classifier_scale > 0 and conf.classifier_path:
            print("loading classifier...")
            classifier = create_classifier(
                **select_args(conf, classifier_defaults().keys())
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(os.path.expanduser(conf.classifier_path),
                                        map_location="cpu")
            )
            classifier.to(device)
            if conf.classifier_use_fp16:
                classifier.convert_to_fp16()
            classifier.eval()
            print("Classifier loaded")

            def cond_fn(x, t, y=None, **_):
                with th.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, t)
                    logp = F.log_softmax(logits, dim=-1)
                    sel = logp[range(len(logits)), y.view(-1)]
                    return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale
        else:
            cond_fn = None
            print("No classifier loaded")

        evaluator = LogitEvaluator(device=device, class_index_path=CLASS_INDEX_PATH)

        def model_fn(x, t, y=None, gt=None, **_):
            return model(x, t, y if conf.class_cond else None, gt=gt)

        pcts = np.arange(0.1, 1.0, 0.1)
        pixels_removed = []
        probabilities   = []
        all_log = []
        logit = []
        images_out      = {}
        masks_out      = {}
        L2_distances = []

        for pct in pcts:
            k = int(conf.image_size * conf.image_size * pct)
            out_knock, mask = generate_knockout(img_np, method_xai, k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float().to(device)
            # opening of mask 
            mask_np = ski.morphology.binary_opening(
                mask.squeeze().cpu().numpy(),
                footprint=ski.morphology.disk(3)
            )

            mask = th.from_numpy(mask_np).float().to(device)    
            mask = mask.unsqueeze(0).unsqueeze(0)

            mask_out_name = f"{img_name}_mask_pct{int(pct*100)}.png"
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            masks_out[mask_out_name] = mask_np
            out_mask_path = os.path.join(conf.output_dir, mask_out_name)
            os.makedirs(conf.output_dir, exist_ok=True)
            mask_pil.save(out_mask_path)

            print(f"Mask opening disk size: {3}, shape: {mask.shape} and saved in {out_mask_path}")

            y_classes = (th.ones(gt.size(0), dtype=th.long, device=device) * conf.cond_y
                     if conf.cond_y is not None else
                     th.randint(0, NUM_CLASSES, (gt.size(0),), device=device))

            model_kwargs = {"gt": gt,
                            "gt_keep_mask": mask,
                            "y": y_classes}

            sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop
            result = sample_fn(
                model_fn,
                (gt.size(0), 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=conf.show_progress,
                return_all=True,
                conf=conf,
            )
            print(result.keys())
            sr = result["sample"]
            gts_tensor = result["gt"]
            # masked image
            lrs_tensor = result["gt"] * mask + (-1) * (1 - mask)


            srs_np = to_uint8(sr)[0]
            gts_np = to_uint8(gts_tensor)[0]
            lrs_np = to_uint8(lrs_tensor)[0]
            img_inpaint_pil = Image.fromarray(srs_np)
            img_gt_pil      = Image.fromarray(gts_np)
            img_lr_pil      = Image.fromarray(lrs_np)

            # P[class]
            logits_gt     = evaluator.evaluate(img_gt_pil)
            idx_gt, lbl_gt = evaluator.predict(logits_gt)
            logits_inp    = evaluator.evaluate(img_inpaint_pil)
            probs_inp = F.softmax(th.from_numpy(logits_inp), dim=0).numpy()
            logits_masked = evaluator.evaluate(img_lr_pil)
            idx_masked, lbl_masked = evaluator.predict(logits_masked)
            print(f"GT class: {lbl_gt} (idx={idx_gt}), Inpainted class: {lbl_masked} (idx={idx_masked})")

            #probs_inp = logits_inp
            p_inp = float(probs_inp[idx_gt])
            #L2 distance
            l2_dist = L2_img(srs_np, gts_np)
            L2_distances.append(l2_dist)

            pixels_removed.append(pct)
            probabilities.append(p_inp)
            all_log.append(logits_inp)
            logit.append(logits_inp[idx_gt])
            
            img_out_name = f"{img_name}_pct{int(pct*100)}.png"
            images_out[img_out_name] = img_inpaint_pil
            out_img_path = os.path.join(conf.output_dir, img_out_name)
            os.makedirs(conf.output_dir, exist_ok=True)
            img_inpaint_pil.save(out_img_path)

            print(f"pct={int(pct*100)}%, prob={p_inp}, logit={logits_inp[idx_gt]}")

        out_data = {
            "image": img_name,
            "pixels_removed": [float(p) for p in pixels_removed],
            "probabilities": [float(p) for p in probabilities],
            "all_log": [lg.tolist() for lg in all_log],
            "logit": [float(l_) for l_ in logit],
            "L2_distances": [float(d) for d in L2_distances],
        }

        out_json_path = os.path.join(conf.output_dir, f"{img_name}_AURC.json")
        os.makedirs(conf.output_dir, exist_ok=True)
        with open(out_json_path, 'w') as f:
            json.dump(out_data, f)
        print("JSON saved to", out_json_path)





        # AURC
        xs = np.array(pixels_removed)
        ys = np.array([p if p is not None else 0 for p in probabilities])
        auc = np.trapz(ys, xs)
        plt.fill_between(xs, ys, alpha=0.4)
        plt.plot(xs, ys)
        plt.xlabel("Pixels removed")
        plt.ylabel(f"P[class={lbl_gt}]")
        plt.title(f"AURC = {auc:.3f}")
        out_plot = os.path.join(conf.output_dir, f"{img_name}_AURC.png")
        plt.savefig(out_plot)
        plt.close()
        print("Plot saved to", out_plot)

        # AURC logits 
        ys_logit = np.array([lg if lg is not None else 0 for lg in logit])
        auc_logit = np.trapz(ys_logit, xs)
        plt.fill_between(xs, ys_logit, alpha=0.4)
        plt.plot(xs, ys_logit)
        plt.xlabel("Pixels removed")
        plt.ylabel(f"Logit[class={lbl_gt}]")
        plt.title(f"AURC Logit = {auc_logit:.3f}")
        out_logit_plot = os.path.join(conf.output_dir, f"{img_name}_AURC_logits.png")
        plt.savefig(out_logit_plot)
        plt.close()
        print("Logit plot saved to", out_logit_plot)

        # L2 distance curve 
        plt.figure(figsize=(8, 6))
        plt.plot(pixels_removed, L2_distances, marker='o', linestyle='-', color='blue')
        plt.xlabel("Pixels removed")
        plt.ylabel("L2 Distance")
        out_l2_plot = os.path.join(conf.output_dir, f"{img_name}_l2_distance.png")
        plt.savefig(out_l2_plot)
        plt.close()
        print("L2 distance plot saved to", out_l2_plot)

        # # mask superposition
        # sorted_items = sorted(
        #     masks_out.items(),
        #     key=lambda kv: int(kv[0].split("pct")[1].split(".")[0])
        # )
        # names, masks = zip(*sorted_items)
        # rings = []
        # for i, m in enumerate(masks):
        #     m_bin = m < 1
        #     if i == 0:
        #         ring = m_bin
        #     else:
        #         prev_bin = masks[i-1] < 1
        #         ring = m_bin & ~prev_bin
        #     rings.append(ring)

        # colors = ["red","blue","green","yellow","magenta","cyan","orange","purple","lime", "indigo", "brown", "pink"]

        # plt.figure(figsize=(8,8))
        # plt.imshow(img_np)

        # legend_handles = []
        # for ring, name, color in zip(rings, names, colors):
        #     masked = np.ma.masked_where(~ring, ring)
        #     cmap = ListedColormap([color])
        #     plt.imshow(masked, cmap=cmap, alpha=0.5)
        #     legend_handles.append(Patch(facecolor=color, edgecolor="k", label=name))

        # plt.axis("off")
        # plt.legend(
        #     handles=legend_handles,
        #     bbox_to_anchor=(1.05,1),
        #     loc="upper left",
        #     borderaxespad=0.
        # )
        # plt.tight_layout()

        # out_overlay = os.path.join(conf.output_dir, f"{img_name}_masks_overlay.png")
        # plt.savefig(out_overlay, dpi=150)
        # plt.close()
        # print("Overlay rings saved to", out_overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, default=None)
    args = parser.parse_args()

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))
    conf_arg.output_dir = conf_arg.get("AURC_output_dir", "./AURC_outputs")
    main(conf_arg)
