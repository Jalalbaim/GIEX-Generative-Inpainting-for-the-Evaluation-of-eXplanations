"""
RePaint AURC / AUDC
@author: J.BAIM
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
import modules.AE_inet as AE_inet
from modules.pixel_wise_comp import count_identical_adjacent

CLASS_INDEX_PATH = "./data/weights/imagenet_class_index.json"
path = "data/weights/resnet50-11ad3fa6.pth"
AE_PATH = "./data/weights/inet_AE.pth"


def load_model(path):
    model = models.resnet50(weights=None)
    state_dict = th.load(path, map_location=th.device("cpu"))
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
    means = th.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    stds = th.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    return t * stds + means


def generate_knockout(img_np, xai, k):
    img_pil = Image.fromarray(img_np)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    prep = transforms.Compose(
        [
            transforms.Resize(img_np.shape[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    inp = prep(img_pil).unsqueeze(0)
    heatmap = SaliencyAttributor(model=load_model(path), method=xai)
    attr = heatmap.compute(inp)
    # masque + knockout
    masker = KnockoutMaskGenerator(K=k, means=means, stds=stds)
    out_knock, mask, _ = masker.generate(inp, attr)
    return out_knock, mask


def L2_img(img1, img2):

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    return np.sqrt(np.sum((img1 - img2) ** 2))


class LogitEvaluator:
    def __init__(self, device: th.device, class_index_path: str):
        self.device = device
        with open(class_index_path, "r") as f:
            idx_map = json.load(f)
        self.idx2label = [idx_map[str(i)][1] for i in range(len(idx_map))]

        self.model = load_model(path)
        self.model.to(self.device)

        self.tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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

def compute_ae_loss(ae, img_pil: Image.Image, device) -> float:
    tf_ae = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    img_tensor = tf_ae(img_pil).unsqueeze(0).to(device)
    with th.no_grad():
        recon = ae(img_tensor)
        loss = F.mse_loss(recon, img_tensor, reduction="mean")
    return loss.item()

def main(conf: conf_mgt.Default_Conf):
    print("Start", conf["name"])
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    print("device:", device)

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
            dist_util.load_state_dict(
                os.path.expanduser(conf.classifier_path), map_location="cpu"
            )
        )
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        print("Classifier loaded")

    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset="eval", dsName=eval_name)
    print("Dataloader size:", len(dl))

    ae = AE_inet.get_AE(AE_PATH, device=device)
    print("AE model loaded")

    for batch in dl:
        for k, v in batch.items():
            if isinstance(v, th.Tensor):
                batch[k] = v.to(device)
        gt = batch["GT"]
        img_name = batch["GT_name"][0].split(".")[0]

        img_np = to_uint8(gt)[0]

        # param XAI
        method_xai = conf.get("method_xai", "saliency")
        print("XAI method:", method_xai)

        if conf.classifier_scale > 0 and conf.classifier_path:
            print("loading classifier...")
            classifier = create_classifier(
                **select_args(conf, classifier_defaults().keys())
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(
                    os.path.expanduser(conf.classifier_path), map_location="cpu"
                )
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

        # init histo 
        pcts = np.arange(0.1, 0.9, 0.1)
        pixels_left = []
        probabilities_inp = []
        probabilities_masked = []
        all_log = []

        L2_distances = []
        logits_repainted = []
        logits_deleted = []
        ae_loss_rep = [] 
        ae_loss_masked = [] 

        inp_identicals = []
        lrs_identicals = []
        gt_identicals = []

        logit_gt = evaluator.evaluate(Image.fromarray(img_np))
        idx_gt, lbl_gt = evaluator.predict(logit_gt)
        print(f"GT class: {lbl_gt} (index {idx_gt})")
        proba_gt = F.softmax(th.from_numpy(logit_gt), dim=0).numpy()[idx_gt]

        ae_loss_gt = compute_ae_loss(ae, Image.fromarray(img_np), device)
        print(f"GT AE loss: {ae_loss_gt:.4f}")

        # probabilities_inp.append(proba_gt)
        # probabilities_masked.append(proba_gt)
        # all_log.append(logit_gt)
        # logits_repainted.append(logit_gt[idx_gt])
        # logits_deleted.append(logit_gt[idx_gt])
        # ae_loss_rep.append(ae_loss_gt)
        # ae_loss_masked.append(ae_loss_gt)

        for pct in pcts:
            k = int(conf.image_size * conf.image_size * pct)
            out_knock, mask = generate_knockout(img_np, method_xai, k)
            mask = 1 - mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float().to(device)
            # opening of mask
            mask_np = mask.squeeze().cpu().numpy()
            mask = th.from_numpy(mask_np).float().to(device)
            mask = mask.unsqueeze(0).unsqueeze(0)

            # blurred input
            smoothed_gt = ski.filters.gaussian(gt.cpu(), sigma=1)
            unmasked_img = smoothed_gt * (1 - mask_np) + (-1) * mask_np
            gt_cpu = gt.cpu()
            masked_img = gt_cpu * mask_np + (-1) * (1 - mask_np)  + unmasked_img

            print(masked_img.shape, gt_cpu.shape, unmasked_img.shape)
            print(masked_img.dtype, gt_cpu.dtype, unmasked_img.dtype)
            # # # plot masked image
            # plt.figure(figsize=(10, 5)) 
            # img_lr_pil = Image.fromarray(masked_img_display)
            # plt.imshow(img_lr_pil)
            # plt.title("Masked Image")
            # plt.axis("off")
            # plt.show()

            y_classes = (
                th.ones(gt.size(0), dtype=th.long, device=device) * conf.cond_y
                if conf.cond_y is not None
                else th.randint(0, NUM_CLASSES, (gt.size(0),), device=device)
            )

            model_kwargs = {"gt": gt, "gt_keep_mask": mask, "y": y_classes}

            sample_fn = (
                diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop
            )
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

            sr = result["sample"]
            gts_tensor = result["gt"]
            lrs_tensor = result["gt"] * mask + (-1) * (1 - mask)

            srs_np = to_uint8(sr)[0]
            gts_np = to_uint8(gts_tensor)[0]
            lrs_np = to_uint8(lrs_tensor)[0]
            img_inpaint_pil = Image.fromarray(srs_np)
            img_gt_pil = Image.fromarray(gts_np)
            img_lr_pil = Image.fromarray(lrs_np)

            # P[class]
            logits_gt = evaluator.evaluate(img_gt_pil)
            idx_gt, lbl_gt = evaluator.predict(logits_gt)
            logits_inp = evaluator.evaluate(img_inpaint_pil)
            logits_masked = evaluator.evaluate(img_lr_pil)
        
            # Logit et proba de la classe GT
            prob_inp = F.softmax(th.from_numpy(logits_inp), dim=0).numpy()[idx_gt]
            prob_masked = F.softmax(th.from_numpy(logits_masked), dim=0).numpy()[idx_gt]

            # L2 distance
            l2_dist = L2_img(srs_np, gts_np)

            # AE loss
            loss_inp = compute_ae_loss(ae, img_inpaint_pil, device)
            loss_masked = compute_ae_loss(ae, img_lr_pil, device)
            ae_loss_rep.append(loss_inp)
            ae_loss_masked.append(loss_masked)

            # Count identical adjacent pixels
            srs_identical = count_identical_adjacent(img_inpaint_pil)
            lrs_identical = count_identical_adjacent(img_lr_pil)
            gt_identical = count_identical_adjacent(img_gt_pil)

            pixels_left.append(pct)
            probabilities_inp.append(prob_inp)
            probabilities_masked.append(prob_masked)
            all_log.append(logits_inp)
            L2_distances.append(l2_dist)
            # ae_loss.append(loss_inp)
            # ae_loss.append(loss_masked)

            logits_repainted.append(logits_inp[idx_gt])
            logits_deleted.append(logits_masked[idx_gt])

            inp_identicals.append(srs_identical)
            lrs_identicals.append(lrs_identical)
            gt_identicals.append(gt_identical)

            # save
            img_out_name = f"{img_name}_pct{int(pct*100)}.png"
            out_img_path = os.path.join(conf.output_dir, img_out_name)
            os.makedirs(conf.output_dir, exist_ok=True)
            img_inpaint_pil.save(out_img_path)

            #save lrs 
            img_out_name_lrs = f"{img_name}_lrs_pct{int(pct*100)}.png"
            out_img_path_lrs = os.path.join(conf.output_dir, img_out_name_lrs)
            os.makedirs(conf.output_dir, exist_ok=True)
            img_lr_pil.save(out_img_path_lrs)

            print(
                f"pct={int(pct*100)}%, prob_inp={prob_inp:.4f}, prob_masked={prob_masked:.4f}, "
                f"logit_repainted={logits_inp[idx_gt]:.4f}, "
                f"logit_deleted={logits_masked[idx_gt]:.4f}"
            )
        probabilities_inp.append(proba_gt)
        probabilities_masked.append(proba_gt)
        all_log.append(logit_gt)
        logits_repainted.append(logit_gt[idx_gt])
        logits_deleted.append(logit_gt[idx_gt])
        ae_loss_rep.append(ae_loss_gt)
        ae_loss_masked.append(ae_loss_gt)
        
        # Save data
        data = {
            "img_name": img_name,
            "pixels_left": pixels_left,
            "probabilities_inp": probabilities_inp,
            "probabilities_masked": probabilities_masked,
            "logits_repainted": logits_repainted,
            "logits_deleted": logits_deleted,
            "L2_distances": L2_distances,
            "ae_loss_rep": ae_loss_rep,
            "ae_loss_masked": ae_loss_masked,
            "inp_identical": inp_identicals,
            "lrs_identical": lrs_identicals,
            "gt_identical": gt_identicals,
        }

        json_out_path = os.path.join(conf.output_dir, f"{img_name}_auic.json")
        os.makedirs(conf.output_dir, exist_ok=True)
        with open(json_out_path, "w") as f:
            json.dump(
                data,
                f,
                indent=4,
                sort_keys=True,
                default=lambda o: o.tolist() if hasattr(o, "tolist") 
                                else float(o) if hasattr(o, "item") 
                                else o
            )

        print(f"Data saved in {json_out_path}")

        # # Plot
        # pixels_left.append(1.0) 
        # x = np.array(pixels_left) * 100

        # y_rep = np.array(probabilities_inp)
        # y_del = np.array(probabilities_masked)

        # y_log_rep = np.array(logits_repainted)
        # y_log_del = np.array(logits_deleted)

        # aurc = np.trapz(y_rep, x)
        # audc = np.trapz(y_del, x)

        # aurc_log = np.trapz(y_log_rep, x)
        # audc_log = np.trapz(y_log_del, x)

        # # AE loss to plot
        # ae_loss_rep = np.array(ae_loss_rep)
        # ae_loss_masked = np.array(ae_loss_masked)

        # # plt.figure()
        # # plt.subplot(1, 2, 1)
        # # plt.plot(x, y_rep, label=f"Repainted (AUIC with RePaint={aurc:.2f})")
        # # plt.plot(x, y_del, "--", label=f"Deleted (AUIC={audc:.2f})")
        # # plt.xlabel("Pixels retirés (%)")
        # # plt.ylabel("proba classe GT")
        # # # add AE loss with a secondary y-axis
        # # ax2 = plt.gca().twinx()
        # # ax2.plot(x, ae_loss_rep, ":", label="AE Loss Repainted")
        # # ax2.plot(x, ae_loss_masked, ":", label="AE Loss Masked")
        # # ax2.set_ylabel("AE Loss")
        # # plt.title("AUIC with RePaint vs AUIC")
        # # plt.legend()
        # # plt.tight_layout()

        # # plt.subplot(1, 2, 2)
        # # plt.plot(x, y_log_rep, label=f"Repainted Logits (AUIC with RePaint={aurc_log:.2f})")
        # # plt.plot(x, y_log_del, "--", label=f"Deleted Logits (AUIC={audc_log:.2f})")
        # # plt.xlabel("Pixels retirés (%)")
        # # plt.ylabel("Logit classe GT")
        # # plt.title("AUIC with RePaint vs AUIC Logits")
        # # ax2 = plt.gca().twinx()
        # # ax2.plot(x, ae_loss_rep, ":", label="AE Loss Repainted")
        # # ax2.plot(x, ae_loss_masked, ":", label="AE Loss Masked")
        # # ax2.set_ylabel("AE Loss")
        # # plt.legend()
        # # plt.tight_layout()
        
        # # out_plot = os.path.join(conf.output_dir, f"{img_name}_aurc_audc.png")
        # # plt.savefig(out_plot, dpi=300)
        # # plt.close()

        # # fig, axes = plt.subplots(1, 2, figsize=(14, 6))


        # # ax1 = axes[0]
        # # lns1 = ax1.plot(x, y_rep,    label=f"Repainted (AUIC_RePaint={aurc:.2f})")
        # # lns2 = ax1.plot(x, y_del, '--', label=f"Deleted (AUIC={audc:.2f})")
        # # ax1.set_xlabel("Pixels left (%)")
        # # ax1.set_ylabel("Probability GT class")
        # # ax1.set_title("AUIC with RePaint vs AUIC")


        # # # ax1b = ax1.twinx()
        # # # lns3 = ax1b.plot(x, ae_loss_rep,   ':', label="AE Loss Repainted", color='red')
        # # # lns4 = ax1b.plot(x, ae_loss_masked, ':', label="AE Loss Masked", color='green')
        # # # ax1b.set_ylabel("AE Loss")

        # # # lns = lns1 + lns2 + lns3 + lns4
        # # # labels = [l.get_label() for l in lns]
        # # # ax1.legend(lns, labels, loc='best')

        # # ax2 = axes[1]
        # # lns1 = ax2.plot(x, y_log_rep,    label=f"Repainted Logits (AUIC_RePaint={aurc_log:.2f})")
        # # lns2 = ax2.plot(x, y_log_del, '--', label=f"Deleted Logits (AUIC={audc_log:.2f})")
        # # ax2.set_xlabel("Pixels left (%)")
        # # ax2.set_ylabel("Logit GT class")
        # # ax2.set_title("AUIC_RePaint Logits vs AUIC Logits")

        # # # ax2b = ax2.twinx()
        # # # lns3 = ax2b.plot(x, ae_loss_rep,   ':',label="AE Loss Repainted", color='red')
        # # # lns4 = ax2b.plot(x, ae_loss_masked, ':', label="AE Loss Masked", color='green')
        # # # ax2b.set_ylabel("AE Loss")

       
        # # # lns = lns1 + lns2 + lns3 + lns4
        # # # labels = [l.get_label() for l in lns]
        # # # ax2.legend(lns, labels, loc='best')

        # # fig.tight_layout()
        # #plt.show()

        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ax1 = axes[0]
        # lns1 = ax1.plot(x, y_rep,    label=f"Repainted (AUIC={aurc:.2f})")
        # lns2 = ax1.plot(x, y_del, '--', label=f"Deleted (AUIC={audc:.2f})")
        # ax1.set_xlabel("Pixels left (%)")
        # ax1.set_ylabel("Probability GT class")
        # ax1.set_title("AUIC with RePaint vs AUIC")
        # lns = lns1 + lns2
        # labels = [l.get_label() for l in lns]
        # ax1.legend(lns, labels, loc='best')

        # ax2 = axes[1]
        # lns1 = ax2.plot(x, y_log_rep,    label=f"Repainted Logits (AURC={aurc_log:.2f})")
        # lns2 = ax2.plot(x, y_log_del, '--', label=f"Deleted Logits (AUDC={audc_log:.2f})")
        # ax2.set_xlabel("Pixels left (%)")
        # ax2.set_ylabel("Logit GT class")
        # ax2.set_title("AUIC with RePaint Logits vs AUIC Logits")
        # lns = lns1 + lns2
        # labels = [l.get_label() for l in lns]
        # ax2.legend(lns, labels, loc='best')

        # fig.tight_layout()
        # #plt.show()

        # out_plot = os.path.join(conf.output_dir, f"{img_name}_auic.png")
        # plt.savefig(out_plot, dpi=300)

        # print(
        #     f"AUIC_R={aurc:.4f}, AUIC={audc:.4f} — courbes sauvées dans {out_plot}\n"
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=True, help="Chemin du fichier YAML de configuration")
    args = parser.parse_args()

    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(args.conf_path))
    main(conf)
