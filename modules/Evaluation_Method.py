"""
This script implements various saliency attribution methods for a pretrained ResNet-50 model.
@author: J.BAIM
"""
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import (
    Saliency,
    IntegratedGradients,
    InputXGradient,
    LayerGradCam,
    GuidedBackprop,
    NoiseTunnel,
    LRP
)

PATH = "data/weights/resnet50-11ad3fa6.pth"

def load_model(path: str):
    model = models.resnet50(weights=None)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

class SaliencyAttributor:
    def __init__(self, model, method: str = "saliency"):
        self.model = model.eval()
        self.method = method.lower()

        self.saliency    = Saliency(model)
        self.int_grad    = IntegratedGradients(model)
        self.inp_x_grad  = InputXGradient(model)
        self.guided_bp   = GuidedBackprop(model)
        self.noise_tn    = NoiseTunnel(self.saliency)
        
        # Detect model type and select appropriate layer for Grad-CAM
        if hasattr(model, 'layer4'):  # ResNet
            target_layer = model.layer4[-1].conv3
        elif hasattr(model, 'features'):  # VGG
            target_layer = model.features[-1]
        else:
            raise ValueError("Unsupported model architecture. Expected ResNet or VGG.")
        
        self.layer_cam   = LayerGradCam(model, target_layer)
        self.lrp         = LRP(model)

    @staticmethod
    def _normalize_0_1(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    @staticmethod
    def _blur_3x3(t):
        return TF.gaussian_blur(t.unsqueeze(0), kernel_size=5, sigma=2).squeeze(0)

    def compute(self, input_tensor, target: int | None = None):
        with torch.no_grad():
            logits = self.model(input_tensor)
        if target is None:
            target = logits.argmax(dim=1).item()

        m = self.method
        if m == "saliency":
            attr = self.saliency.attribute(input_tensor, target=target)

        elif m == "integrated gradients":
            attr = self.int_grad.attribute(input_tensor, target=target, n_steps=50)

        elif m == "grad-cam":
            cam = self.layer_cam.attribute(input_tensor, target=target)          
            cam = cam.mean(dim=1, keepdim=True)                                  
            cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
            heat = cam.squeeze().cpu()
            return self._normalize_0_1(heat).detach().numpy()

        elif m == "grads":
            attr = self.inp_x_grad.attribute(input_tensor, target=target)

        elif m == "smoothgrad":
            attr = self.noise_tn.attribute(input_tensor, nt_type="smoothgrad", target=target, nt_samples=50)

        elif m == "gbp":
            attr = self.guided_bp.attribute(input_tensor, target=target)
        
        elif m == "lrp":
            attr = self.lrp.attribute(input_tensor, target=target)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # |attr| et somme des canaux
        heat = torch.abs(attr[0]).sum(dim=0)

        # post-processing
        if m in {"saliency", "integrated gradients", "grads", "smoothgrad", "gbp", "lrp"}:
            heat = self._blur_3x3(heat)
        heat = self._normalize_0_1(heat)

        return heat.detach().cpu().numpy()


def main():
    img_path = "data/eagle_inet/ILSVRC2012_val_00021056.jpeg"
    img = Image.open(img_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    inp = preprocess(img).unsqueeze(0)

    model = load_model(PATH)

    methods = [
        "saliency",
        "integrated gradients",
        "grad-cam",
        "grads",
        "smoothgrad",
        "gbp"
    ]

    heatmaps = {}
    for m in methods:
        heatmaps[m] = SaliencyAttributor(model, m).compute(inp)

    n = len(methods) + 1
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))

    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, m in enumerate(methods, start=1):
        axs[i].imshow(heatmaps[m], cmap="hot")
        title = {
            "grad-cam": "Grad-CAM",
            "grads": "InputXGrad",
            "smoothgrad": "SmoothGrad",
            "gbp": "GBP"
        }.get(m, m.title())
        axs[i].set_title(title)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
