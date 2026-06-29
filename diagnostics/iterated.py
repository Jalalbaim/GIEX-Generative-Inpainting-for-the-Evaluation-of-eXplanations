"""
This script implements an iterative pipeline for image inpainting using RePaint.
@author: J.BAIM
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import skimage as ski

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
from modules.knockout import KnockoutMaskGenerator
import modules.AE_inet as AE_inet


MIN_K = 13_000
MAX_ITER = 5
AE_PATH = "./data/weights/inet_AE.pth"
CLASS_INDEX_PATH = "./data/weights/imagenet_class_index.json"
RESNET_PATH = "data/weights/resnet50-11ad3fa6.pth"


def load_model(path: str) -> th.nn.Module:
    """Charge un ResNet‑50 ImageNet en évaluation (CPU par défaut)."""
    model = models.resnet50(weights=None)
    state_dict = th.load(path, map_location=th.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def to_uint8(t: th.Tensor):
    if t is None:
        return None
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def mask_to_uint8(t: th.Tensor):
    return (t.squeeze().cpu().numpy() * 255).astype(np.uint8)


def imagenet_denorm(t: th.Tensor):
    means = th.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    stds = th.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    return t * stds + means


def classify_image_uint8(img_uint8: np.ndarray, classifier: th.nn.Module, device: th.device) -> int:
    pil = Image.fromarray(img_uint8)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with th.no_grad():
        logits = classifier(tensor)
    pred = logits.softmax(dim=1).argmax(dim=1).item()
    return pred


def generate_knockout(img_np: np.ndarray, xai: str, k: int):
    """Génère (out_knock, mask, attr) à partir d’une image np.uint8."""
    img_pil = Image.fromarray(img_np)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ])
    inp = prep(img_pil).unsqueeze(0)
    model = SharedClassifier.instance().to(inp.device)
    heatmap = SaliencyAttributor(model=model, method=xai)
    attr = heatmap.compute(inp)

    masker = KnockoutMaskGenerator(K=k, means=means, stds=stds)
    out_knock, mask, _ = masker.generate(inp, attr)
    return out_knock, mask, attr


class SharedClassifier:
    _model: th.nn.Module | None = None
    _device: th.device | None = None

    @classmethod
    def instance(cls, device: th.device | None = None):
        if cls._model is None:
            cls._model = load_model(RESNET_PATH)
            cls._device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
            cls._model.to(cls._device)
        elif device is not None and device != cls._device:
            # déplace si pipeline tourne sur un autre GPU/CPU
            cls._model.to(device)
            cls._device = device
        return cls._model

    @classmethod
    def device(cls):
        return cls._device or th.device("cpu")


class EvaluatorPhase2:

    def __init__(
        self,
        gt_dir: str,
        masked_dir: str,
        inpaint_dir: str,
        output_dir: str,
        report_path: str,
        xai: str,
        device: th.device,
    ):
        self.gt_dir = Path(gt_dir)
        self.masked_dir = Path(masked_dir)
        self.inpaint_dir = Path(inpaint_dir)
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path)
        self.xai = xai
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = SharedClassifier.instance(device)
        with open(CLASS_INDEX_PATH, "r") as f:
            idx_json = json.load(f)
        self.idx2label = [idx_json[str(i)][1] for i in range(len(idx_json))]

        # Auto‑encodeur ImageNet pour AE‑loss
        self.ae = AE_inet.get_AE(AE_PATH, self.device)
        self.tf_ae = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])


    def _preprocess(self, img: Image.Image) -> th.Tensor:
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return tf(img).unsqueeze(0)

    def _logits(self, img: Image.Image) -> np.ndarray:
        t = self._preprocess(img).to(self.device)
        self.classifier = self.classifier.to(self.device)
        with th.no_grad():
            logits = self.classifier(t)
        return logits.cpu().numpy().squeeze()

    def _predict(self, logits: np.ndarray):
        idx = int(np.argmax(logits))
        return idx, self.idx2label[idx]

    def run(self):
        from types import SimpleNamespace

        filenames = sorted([f for f in os.listdir(self.gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not filenames:
            raise RuntimeError(f"Aucune image trouvée dans {self.gt_dir}")

        results: dict[str, dict] = {}
        ae_losses: dict[str, dict] = {}

        for name in filenames:
            results[name] = {}
            paths = {
                "GT": self.gt_dir / name,
                "GT_masked": self.masked_dir / name,
                "Inpainted": self.inpaint_dir / name,
            }
            imgs = {lbl: Image.open(p).convert("RGB") for lbl, p in paths.items()}

            logits_dict, pred_dict, cls_dict = {}, {}, {}
            for lbl, img in imgs.items():
                logits = self._logits(img)
                idx, clsname = self._predict(logits)
                logits_dict[lbl] = logits.tolist()
                pred_dict[lbl] = idx
                cls_dict[lbl] = clsname

                t = self.tf_ae(img).unsqueeze(0).to(self.device)
                with th.no_grad():
                    recon = self.ae(t)
                    loss = float(F.mse_loss(recon, t).item())
                ae_losses.setdefault(name, {})[lbl] = loss

            # métriques drop / increase
            logits_gt = np.array(logits_dict["GT"])
            probs_gt = F.softmax(th.from_numpy(logits_gt), dim=0)
            idx_gt = pred_dict["GT"]
            p_orig = float(probs_gt[idx_gt])

            def _avg_drop(p_old, p_new):
                return max(0.0, (p_old - p_new) / p_old) * 100.0

            def _pct_inc(p_old, p_new):
                return 0.0 if p_new <= p_old + 1e-4 else abs(p_new - p_old) * 100.0 / p_old

            metrics = {}
            for lbl in ("GT_masked", "Inpainted"):
                logits_new = np.array(logits_dict[lbl])
                p_new = float(F.softmax(th.from_numpy(logits_new), dim=0)[idx_gt])
                metrics[lbl] = {
                    "average_drop": round(_avg_drop(p_orig, p_new), 4),
                    "percent_increase": int(_pct_inc(p_orig, p_new)),
                }

            results[name] = {
                "logits": logits_dict,
                "pred": pred_dict,
                "class_name": cls_dict,
                "AE_loss": ae_losses[name],
                "metrics": metrics,
            }

        # Sauvegarde JSON
        json_path = self.output_dir / "results_phase2.json"
        with open(json_path, "w") as fp:
            json.dump(results, fp, indent=4)
        print(f"[Phase2] JSON écrit → {json_path}")

        # Rapport PDF
        self._generate_report(results, ae_losses)
        print(f"[Phase2] Rapport PDF écrit → {self.report_path}")

    def _generate_report(self, results: dict, ae_losses: dict):
        n_cases = len(results)
        n_cols = 4
        n_rows = (n_cases * 8 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
        axes = axes.flatten()
        ax_idx = 0

        def _compute_heatmap(img: Image.Image):
            t = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(img).unsqueeze(0)
            t = t.to(self.device)
            model = SharedClassifier.instance(self.device) 
            model = model.to(self.device)
            attributor = SaliencyAttributor(model, method=self.xai)
            return attributor.compute(t)

        for name, info in results.items():
            imgs = {
                "GT (orig)": Image.open(self.gt_dir / name).convert("RGB"),
                "GT_masked": Image.open(self.masked_dir / name).convert("RGB"),
                "Inpainted": Image.open(self.inpaint_dir / name).convert("RGB"),
            }
            # images & heatmaps
            for title, img in imgs.items():
                for _ in (0, 1):
                    ax = axes[ax_idx]
                    ax.axis("off")
                    if _ == 0:
                        ax.imshow(img)
                        ax.set_title(title)
                    else:
                        ax.imshow(_compute_heatmap(img), cmap="hot")
                        ax.set_title(f"Heatmap {title}")
                    ax_idx += 1
            # logits barplot
            ax = axes[ax_idx]
            idx_gt = info["pred"]["GT"]
            bars = [
                F.softmax(th.tensor(info["logits"][lbl]), dim=0)[idx_gt].item()
                for lbl in ("GT", "GT_masked", "Inpainted")
            ]
            ax.bar(["GT", "GT_masked", "Inpainted"], bars)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilité softmax")
            ax.set_title(f"Classe « {info['class_name']['GT']} »")
            ax_idx += 1
            # text zone
            ax = axes[ax_idx]
            ax.axis("off")
            txt_lines = [
                f"AE loss GT        : {info['AE_loss']['GT']:.4f}",
                f"AE loss Masked    : {info['AE_loss']['GT_masked']:.4f}",
                f"AE loss Inpainted : {info['AE_loss']['Inpainted']:.4f}",
                "",
                f"Avg drop Masked   : {info['metrics']['GT_masked']['average_drop']:.2f}%",
                f"% increase Masked : {info['metrics']['GT_masked']['percent_increase']}%",
                f"Avg drop Inpaint  : {info['metrics']['Inpainted']['average_drop']:.2f}%",
                f"% increase Inpaint: {info['metrics']['Inpainted']['percent_increase']}%",
            ]
            ax.text(0.05, 0.95, "\n".join(txt_lines), va="top", ha="left", fontsize=9, family="monospace")
            ax_idx += 1

        for j in range(ax_idx, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(self.report_path, dpi=300)
        plt.close(fig)



def run_iterative_pipeline(conf: conf_mgt.Default_Conf):
    print("Start:", conf["name"])
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
    print("Modèle de diffusion chargé.")

    # Classifier guidance
    if conf.classifier_scale > 0 and conf.classifier_path:
        print("Chargement du classifieur pour guidance…")
        classifier_guidance = create_classifier(
            **select_args(conf, classifier_defaults().keys())
        )
        classifier_guidance.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        classifier_guidance.to(device)
        if conf.classifier_use_fp16:
            classifier_guidance.convert_to_fp16()
        classifier_guidance.eval()
        print("Classifieur guidance chargé.")

        def cond_fn(x, t, y=None, **_):
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier_guidance(x_in, t)
                logp = F.log_softmax(logits, dim=-1)
                sel = logp[range(len(logits)), y.view(-1)]
                return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None
        print("Pas de classifieur de guidance (cond_fn=None).")

    print("Chargement de ResNet50 pour vérification de classe…")
    classifier_check = SharedClassifier.instance(device)
    print("ResNet50 pour vérification chargé.")

    def model_fn(x, t, y=None, gt=None, **_):
        return model(x, t, y if conf.class_cond else None, gt=gt)


    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset="eval", dsName=eval_name)
    print("Taille du DataLoader :", len(dl))

    for batch in dl:

        for k, v in batch.items():
            if isinstance(v, th.Tensor):
                batch[k] = v.to(device)
        gt = batch["GT"]
        batch_size = gt.shape[0]
        if batch_size != 1:
            raise ValueError("Ce script nécessite batch_size == 1 pour traitement itératif.")

        img_name_full = batch["GT_name"][0]
        img_name = os.path.splitext(img_name_full)[0]
        print(f"\n— Traitement de l’image : {img_name_full} —")

        img_np_gt = to_uint8(gt)[0]
        GT_class = classify_image_uint8(img_np_gt, classifier_check, device)
        print(f"Classe d’origine (GT_class) : {GT_class}")

        path_conf = conf["data"]["eval"][eval_name]["paths"]
        srs_path = path_conf["srs"]
        logits_root = path_conf["logits"]
        parent_srs = os.path.dirname(srs_path)
        iter_root = os.path.join(parent_srs, "iter_results", img_name)
        os.makedirs(iter_root, exist_ok=True)
        print(f"Dossier de sauvegarde des itérations : {iter_root}")

        iter_count, class_pred = 0, GT_class
        current_gt = gt.clone()
        current_img_np = img_np_gt.copy()

        while iter_count < MAX_ITER and class_pred == GT_class:
            iter_index = iter_count + 1
            print(f"\nItération {iter_index} / {MAX_ITER}")

            K = MIN_K  # Otsu désactivé pour reproductibilité
            out_knock, mask, attr = generate_knockout(current_img_np, xai=conf.get("method_xai", "saliency"), k=K)

            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float().to(device)

            # ouverture morphologique
            mask_np = ski.morphology.binary_opening(
                mask.squeeze().cpu().numpy(),
                footprint=ski.morphology.disk(3),
            )
            mask = th.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

            # DEBUG VISUEL (inchangé)
            if conf.get("debug_vis", False):
                vis = imagenet_denorm(out_knock.to(device))
                vis_uint8 = (vis * 255).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1)[0].cpu().numpy()
                plt.subplot(1, 2, 1)
                plt.imshow(vis_uint8)
                plt.title("out_knock")
                plt.subplot(1, 2, 2)
                plt.imshow(mask_to_uint8(mask), cmap="gray")
                plt.title("mask")
                plt.show()

            # y_classes pour guidance
            if conf.cond_y is not None:
                y_classes = th.ones(batch_size, dtype=th.long, device=device) * conf.cond_y
            else:
                y_classes = th.randint(0, NUM_CLASSES, (batch_size,), device=device)

            model_kwargs = {"gt": current_gt, "gt_keep_mask": mask, "y": y_classes}
            sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop
            result = sample_fn(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=conf.show_progress,
                return_all=True,
                conf=conf,
            )

            srs_tensor = result["sample"]
            gts_tensor = result["gt"]
            lrs_tensor = result["gt"] * mask + (-1) * (1 - mask)

            srs_np, gts_np = to_uint8(srs_tensor)[0], to_uint8(gts_tensor)[0]
            lrs_np, mask_np_uint8 = to_uint8(lrs_tensor)[0], mask_to_uint8(mask)

            iter_dir = os.path.join(iter_root, f"iter_{iter_index}")
            os.makedirs(iter_dir, exist_ok=True)
            inpaint_iter_dir = os.path.join(iter_dir, "srs")
            masked_iter_dir = os.path.join(iter_dir, "lrs")
            gt_iter_dir = os.path.join(iter_dir, "gts")
            for d in (inpaint_iter_dir, masked_iter_dir, gt_iter_dir):
                os.makedirs(d, exist_ok=True)

            base_name = img_name_full
            Image.fromarray(srs_np).save(os.path.join(inpaint_iter_dir, base_name))
            Image.fromarray(gts_np).save(os.path.join(gt_iter_dir, base_name))
            Image.fromarray(lrs_np).save(os.path.join(masked_iter_dir, base_name))
            Image.fromarray(mask_np_uint8).save(os.path.join(iter_dir, f"{img_name}_mask_iter{iter_index}.png"))

            print(f"→ Résultats sauvegardés dans {iter_dir}")

            logits_dir = os.path.join(logits_root, img_name, f"iter_{iter_index}")
            pdf_dir = logits_dir + "_report"
            os.makedirs(logits_dir, exist_ok=True)

            evaluator = EvaluatorPhase2(
                gt_dir=gt_iter_dir,
                masked_dir=masked_iter_dir,
                inpaint_dir=inpaint_iter_dir,
                output_dir=logits_dir,
                report_path=pdf_dir,
                xai=conf.get("method_xai", "saliency"),
                device=device,
            )
            evaluator.run()
            print(f"Phase2 terminée pour itération {iter_index}.")

            class_pred = classify_image_uint8(srs_np, classifier_check, device)
            print(f"Classe prédite sur srs (itération {iter_index}) : {class_pred}")

            iter_count += 1
            if class_pred != GT_class:
                print("→ Classe différente détectée ; on arrête les itérations.")
                break
            else:
                if iter_count < MAX_ITER:
                    print("→ Même classe ; on passe à l’itération suivante.")
                    current_gt = srs_tensor.detach().clone()
                    current_img_np = srs_np.copy()
                else:
                    print("→ Atteint max_iter ; on arrête les itérations.")

        print(f"\nFin du traitement de {img_name_full} après {iter_count} itération(s).")
        print(f"Classe finale prédite : {class_pred}, Classe d'origine : {GT_class}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=True, help="Chemin du fichier YAML de configuration")
    args = parser.parse_args()

    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(args.conf_path))
    run_iterative_pipeline(conf)
