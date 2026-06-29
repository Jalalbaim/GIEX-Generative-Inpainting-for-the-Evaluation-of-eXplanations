import numpy as np
from PIL import Image
import numpy as np, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.AE_inet import get_AE
from torchvision import transforms
import torch as th
import torch.nn as nn



def shannon_entropy(vec):
    counts = np.bincount(vec, minlength=256).astype(np.float64)
    p = counts[counts > 1] / vec.size
    return -np.sum(p * np.log2(p))

def global_entropy(img):
    return shannon_entropy(img.reshape(-1))


def binary_mask(arr, thr=5):
    return (arr < thr).astype(np.uint8)

def entropy_map(mask, win=8, stride=8):
    h, w   = mask.shape
    oh, ow = (h-win)//stride+1, (w-win)//stride+1
    H      = np.zeros((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            blk = mask[i*stride:i*stride+win, j*stride:j*stride+win]
            p   = blk.mean()
            if p not in (0, 1):
                H[i, j] = -(p*math.log2(p) + (1-p)*math.log2(1-p))
    return H

def analyse(path, thr=5, win=8, stride=8):
    arr   = np.array(Image.open(path).convert('L'))
    mask  = binary_mask(arr, thr)
    Hmap  = entropy_map(mask, win, stride)
    return {
        "file"        : os.path.basename(path),
        "black_pixels": int(mask.sum()),
        "mean_entropy": float(Hmap.mean()),
        "max_entropy" : float(Hmap.max())
    }

def loss_ae(path, device):
    ae = get_AE("data/weights/inet_AE.pth", device=device)
    tf_ae = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
    img = Image.open(path).convert('RGB')
    img = tf_ae(img).unsqueeze(0).to(device)
    with th.no_grad():
        out = ae(img)
        criterion = nn.MSELoss()
        loss = criterion(out, img)
    return loss.item()


def loss_entropy(path, device ,win=8, stride=8):
    arr   = np.array(Image.open(path).convert('L'))
    mask  = binary_mask(arr, thr=5)
    mean_H  = float(entropy_map(mask, win, stride).mean())
    ae_loss = loss_ae(path, device)
    print(f"Mean entropy: {mean_H}, AE loss: {ae_loss}")
    loss = 0.5*mean_H + 0.5*ae_loss
    return loss



if __name__ == "__main__":

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Using device:", device)

    path_img = "logs/water_ou_entropy_gradcam/gt_masked/water_ouzel_img.png"
    path_img2 = "logs/water_ou_entropy_integrated_gradients/gt_masked/water_ouzel_img.png"
    repaint_img = "logs/water_ou_entropy_gradcam/inpainted/water_ouzel_img.png"
    repaint_img2 = "logs/water_ou_entropy_integrated_gradients/inpainted/water_ouzel_img.png"



    print("entropie conditional")
    res1 = analyse(path_img)
    loss_ent = loss_entropy(path_img, device)
    print("gradcam masked")
    print(res1)
    print("loss total gradcam_masked:", loss_ent)
    print("gradcam repaint")
    res1_repaint = analyse(repaint_img)
    loss_ent_repaint = loss_entropy(repaint_img, device)
    print(res1_repaint)
    print("loss total gradcam_repaint:", loss_ent_repaint)

    res2 = analyse(path_img2)
    print("integrated gradients masked")
    print(res2)
    loss_ent2 = loss_entropy(path_img2, device)
    print("loss total integrated_gradients_masked:", loss_ent2)
    print("integrated gradients repaint")
    res2_repaint = analyse(repaint_img2)
    print(res2_repaint)
    loss_ent2_repaint = loss_entropy(repaint_img2, device)
    print("loss total integrated_gradients_repaint:", loss_ent2_repaint)

    # plot imgs, with values 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(Image.open(path_img).convert('RGB'), cmap='gray')    
    ax[0].set_title(f'Original: {res1}')
    ax[1].imshow(Image.open(repaint_img).convert('RGB'), cmap='gray')
    ax[1].set_title('Repainted Image')
    plt.show()





    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # folder = "./logs/water_ouzel_aurc_gradcam"
    # pct = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # loss_rep = []
    # loss_mask = []

    # for p in pct:
    #     print(f"Processing percentage: {p}%")
    #     img_rep = os.path.join(folder, f"water_ouzel_img_pct{p}.png")
    #     img_mask = os.path.join(folder, f"water_ouzel_img_mask_pct{p}.png")
    #     print("Repainted image:", img_rep)
    #     print("Masked image:", img_mask)

    #     total_loss_rep = loss_entropy(img_rep, device)
    #     total_loss_mask = loss_entropy(img_mask, device)

    #     print(f"Total loss for repaint image at {p}%: {total_loss_rep}")
    #     print(f"Total loss for masked image at {p}%: {total_loss_mask}")

    #     loss_rep.append(total_loss_rep)
    #     loss_mask.append(total_loss_mask)

    
    # import matplotlib.pyplot as plt 
    # plt.figure(figsize=(10, 5))
    # plt.plot(pct, loss_rep, label='Repainted Image Loss')
    # plt.plot(pct, loss_mask, label='Masked Image Loss')
    # plt.xlabel('pixels removed')
    # plt.ylabel('Combined Loss')
    # plt.title("GradCam")
    # plt.legend()
    # plt.show()