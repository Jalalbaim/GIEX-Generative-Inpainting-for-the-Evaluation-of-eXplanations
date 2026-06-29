"""
This module defines the KnockoutMaskGenerator class, which generates knockout masks
for input tensors based on attribution scores. It includes methods for denormalizing
the input tensor and generating the knockout mask by zeroing out the top K pixels
based on the attribution scores.

@author: J.BAIM
"""
import torch
from torchvision.transforms import GaussianBlur

class KnockoutMaskGenerator:
    def __init__(self, K, blur_kernel=7, blur_sigma=2.0, means=None, stds=None):
        self.K = K
        self.blur = GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        self.means = means
        self.stds = stds

    @staticmethod
    def denormalize(tensor, mean, std):
        t = tensor.clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            t[0, c] = t[0, c] * s + m
        return t

    def generate(self, input_tensor, attribution):
        smoothed = torch.tensor(attribution) #np array 
        flat = smoothed.flatten()
        idx = torch.topk(flat, self.K).indices
        # mask 1/0
        mask = torch.ones_like(smoothed)
        mask.view(-1)[idx] = 0.0

        out = self.denormalize(input_tensor, self.means, self.stds).clone()
        H, W = smoothed.shape
        for i in idx:
            y, x = divmod(i.item(), W)
            out[0, :, y, x] = 0.0
        out = torch.clamp(out, 0.0, 1.0)
        return out, mask, smoothed
