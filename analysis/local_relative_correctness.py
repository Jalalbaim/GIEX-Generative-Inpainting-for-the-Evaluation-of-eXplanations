"""
@Author: Romain Xu-Darme
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from modules.Evaluation_Method import SaliencyAttributor
import glob

PATH = "./data/weights/resnet50-11ad3fa6.pth"

def load_model(path):
    model = models.resnet50(weights=None)
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    print("Model loaded")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def local_relative_correctness(
        model: nn.Module,
        target: int,
        reference: torch.Tensor,
        saliency_map: torch.Tensor,
        num_samples: int,
        sigma: float,
        stability_factor: float = 1e-5,
        device: str | torch.device = "cuda:0"
) -> float:
    """
    Returns the local relative correctness of a saliency map computed in the neighbourhood of a reference point.

    Args:
        model: Source model to be explained.
        target: Target class of the saliency map.
        reference: Reference point for the analysis (original image preprocessed for the model).
        saliency_map: Saliency map computed from the reference image, the model and the target.
        num_samples: Number of samples for the analysis.
        sigma: Standard deviation of the gaussian noise used to create perturbed samples.
        stability_factor: Stability factor. Default: 1e-5.
        device: Hardware device.
    """
    assert num_samples > 1 and stability_factor > 0

    if reference.ndim == 3:
        reference = reference[None]

    def evaluate_surrogate(Ik: torch.Tensor, I0: torch.Tensor, S0: torch.Tensor, f0: float) -> float:
        """Evaluates a surrogate model at Ik.

        Args:
            Ik: Surrogate input
            I0: Surrogate reference image
            S0: Saliency map for reference image.
            f0: Class logit for reference image
        Returns:
            S0 x (Ik - I0) + f0
        """
        return torch.sum(torch.multiply(Ik - I0, S0)).item() + f0

    avg_error = 0.0

    # Compute reference logit
    model.eval()
    with torch.no_grad():
        reference = reference.to(device)
        saliency_map = saliency_map.to(device)
        F0 = model(reference).cpu()[0, target].item()

        for _ in range(num_samples):
            noise = torch.randn(reference.size()) * (sigma ** 0.5)
            Ik = reference + noise.to(device)
            Fk = model(Ik).cpu()[0, target].item()
            surrogate_estimation = evaluate_surrogate(Ik, reference, saliency_map, F0)
            error = abs(surrogate_estimation - Fk) / (
                    abs(F0 - Fk) + stability_factor)
            avg_error += error

    return avg_error / num_samples


def compute_LRC(img):
    """
    Args:
        img_path: Path to the input image.
    Returns:
        LRC value.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(PATH).to(device)
    methods = [
        # "saliency",
        # "integrated gradients",
        # "grad-cam",
        # "grads",
        # "smoothgrad",
        # "gbp",
        "lrp"
    ]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # predict class
    with torch.no_grad():
        img_class = model(input_tensor).argmax(dim=1).item()
    print(f"Image class: {img_class}")

    LRC={}

    for method in methods:
        attributor = SaliencyAttributor(model=model, method=method)
        sal = attributor.compute(input_tensor, target=img_class)
        # ensure tensor type
        if isinstance(sal, np.ndarray):
            sal = torch.from_numpy(sal)
        if sal.dim() == 2: # Grad Cam 2D to 3 channel
            sal = sal.unsqueeze(0).unsqueeze(0)
        elif sal.dim() == 3 and sal.shape[0] != input_tensor.shape[1]:
            sal = sal.unsqueeze(0)
        sal = sal.to(device)
        lrc = local_relative_correctness(
            model=model,
            target=img_class,
            reference=input_tensor,
            saliency_map=sal,
            num_samples=50,
            sigma=0.3,
            stability_factor=1e-7,
            device=device
        )
        print(f"{method}: {lrc}")
        LRC[method] = lrc
    
    return LRC




if __name__ == "__main__":
    import pandas as pd

    folder = "./data/imagenette_sub15"
    results = {}

    for img_path in glob.glob(f"{folder}/*.JPEG"):
        print(f"Processing {img_path}")
        img = Image.open(img_path).convert("RGB")
        lrc_values = compute_LRC(img)  # lrc_values est un dict : {method: value}

        img_id = img_path.split("/")[-1].split(".")[0]
        print(f"LRC for {img_id}: {lrc_values}")
        results[img_id] = lrc_values

    df = pd.DataFrame.from_dict(results, orient='index')
    df.loc["mean"] = df.mean(numeric_only=True)
    df.to_csv('./results/lrc_results.csv', index_label='image_name')

    print("Results saved to lrc_results.csv")
    print(df.tail())

