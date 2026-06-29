"""
Here we'll use OTSU to compute the K nb of pixels to knockout.
@author: J.BAIM
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch as th
from skimage.filters import threshold_otsu
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.Evaluation_Method import SaliencyAttributor
import torchvision.models as models


PATH = "data/weights/resnet50-11ad3fa6.pth"

def load_model(path):
    model = models.resnet50(weights=None)
    state_dict = th.load(path, map_location=th.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def otsu_k(img_path, xai="saliency" ,verbose=False):
    if isinstance(img_path, str):
        img_pil = Image.open(img_path).convert("RGB")
    elif isinstance(img_path, th.Tensor):
        if img_path.dim() == 4:
            img_path = img_path.squeeze(0)
        img_np = img_path.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ])

    inp = prep(img_pil).unsqueeze(0)

    xai_method = SaliencyAttributor(
        model=load_model(PATH),
        method=xai
    )

    attr = xai_method.compute(inp)
    
    
    # thresholding
    thresh = threshold_otsu(attr)
    print(f"Threshold: {thresh}")
    binary = attr > thresh

    # compute number of pixels to knockout 
    K = np.sum(binary).item()

    if verbose:
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        fig.suptitle(f"Nb pixels K = {K}")
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3)

        ax[0].imshow(attr, cmap=plt.cm.gray)
        ax[0].set_title('Original saliency')
        ax[0].axis('off')

        ax[1].hist(attr.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(thresh, color='r')

        ax[2].imshow(binary, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

    return K


if __name__ == "__main__":
    img = "./data/imgs/original.jpg"
    K = otsu_k(img, verbose=True)
    print(f"Number of pixels to knockout: {K}")
    
