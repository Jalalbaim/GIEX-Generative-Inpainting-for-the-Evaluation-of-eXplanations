# Weights and Data Setup

All model weights and the dataset are gitignored and must be downloaded separately.
Place every file exactly at the path shown — scripts use hardcoded relative paths from the repo root.

---

## Directory layout expected at runtime

```
data/
├── weights/
│   ├── resnet50-11ad3fa6.pth
│   ├── vgg16-397923af.pth
│   ├── 256x256_diffusion.pt
│   ├── 256x256_classifier.pt
│   ├── inet_AE.pth
│   ├── inception_v3_google-0cc3c7bd.pth
│   └── imagenet_class_index.json
└── imagenette_sub15/
    └── <class folders with JPEG images>

deepfillv2-pytorch/
└── pretrained/
    ├── states_pt_places2.pth   (used by FID.py)
    └── states_tf_places2.pth   (used by halluc.py)
```

---

## File-by-file instructions

### `data/weights/resnet50-11ad3fa6.pth`

**Used by:** `Evaluation_Method.py`, `AUIC.py`, `AUDC.py`, `local_relative_correctness.py`, `OTSU.py`, `iterated.py`, `AE_inet.py`

Standard torchvision ResNet-50 pretrained on ImageNet-1k.

```python
import torch, torchvision.models as models
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
torch.save(m.state_dict(), "data/weights/resnet50-11ad3fa6.pth")
```

The filename hash must match exactly (it is the SHA256 prefix used by torchvision).

---

### `data/weights/vgg16-397923af.pth`

**Used by:** `AURC.py`, `FID.py`, `halluc.py`, `deepfill_metrics.py`

Standard torchvision VGG-16 pretrained on ImageNet-1k.

```python
import torch, torchvision.models as models
m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
torch.save(m.state_dict(), "data/weights/vgg16-397923af.pth")
```

---

### `data/weights/inception_v3_google-0cc3c7bd.pth`

**Used by:** `FID.py` only (Inception feature extractor for FID score)

Standard torchvision Inception v3 pretrained on ImageNet-1k.

```python
import torch
from torchvision.models import inception_v3, Inception_V3_Weights
m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
torch.save(m.state_dict(), "data/weights/inception_v3_google-0cc3c7bd.pth")
```

---

### `data/weights/256x256_diffusion.pt` and `data/weights/256x256_classifier.pt`

**Used by:** `AURC.py`, `AUIC.py`, `AUDC.py`, `FID.py`, `halluc.py`, `iterated.py` (via YAML configs)

These are the RePaint diffusion model and guidance classifier, originally released by OpenAI.
Download from the RePaint repository:

- Repository: **https://github.com/andreas128/RePaint**  
  (TODO: confirm exact commit hash used — run `git -C guided_diffusion/ log -1` after cloning)
- Direct download links are provided in that repo's README under "Pretrained models".
- Place `256x256_diffusion.pt` and `256x256_classifier.pt` in `data/weights/`.

---

### `data/weights/inet_AE.pth`

**Used by:** `AURC.py`, `AUIC.py`, `entropy.py`, `iterated.py`

Custom ResNet-50 autoencoder trained on ImageNet-mini to measure in-distribution proximity.
This checkpoint is produced by `AE_inet.py` (see training loop in `__main__`).

```bash
# Train on ImageNet-mini (adjust data_path to your ImageNet-mini location):
python AE_inet.py   # see __main__ block for the data_path variable to set
# Checkpoint saved to AE_imagenet_.pth — rename/move:
mv AE_imagenet_.pth data/weights/inet_AE.pth
```

---

### `data/weights/imagenet_class_index.json`

**Used by:** `AURC.py`, `AUIC.py`, `AUDC.py`, `deepfill_metrics.py`, `iterated.py`

Standard ImageNet class index mapping integer index → synset ID + label.
Available from torchvision's GitHub or any standard ImageNet resource:

```bash
# One-liner:
python -c "import torchvision; import json, urllib.request; \
  url='https://raw.githubusercontent.com/pytorch/vision/main/torchvision/data/imagenet_classes.txt'; \
  # OR download from torchvision source directly"
```

Alternatively, the file is included in many Hugging Face datasets and torchvision examples.
The expected format is `{"0": ["n01440764", "tench"], "1": [...], ...}`.

---

### `data/imagenette_sub15/`

**Used by:** all YAML configs, `deepfill_metrics.py`, `local_relative_correctness.py`, `FID.py`

A 15-class subset of Imagenette (itself a 10-class ImageNet subset), used as the evaluation dataset.

- Imagenette source: **https://github.com/fastai/imagenette**
- Download Imagenette-160 or Imagenette-320 and place images under `data/imagenette_sub15/`.
- TODO: document exact subset selection / class list used in the paper.

---

## External code packages (not vendored)

### `guided_diffusion/`

**Used by:** `AURC.py`, `AUIC.py`, `AUDC.py`, `FID.py`, `halluc.py`, `iterated.py`, `conf_mgt/conf_base.py`

RePaint's guided-diffusion package. Clone into the repo root:

```bash
git clone https://github.com/andreas128/RePaint
# Then copy or symlink the guided_diffusion/ subfolder to the repo root:
cp -r RePaint/guided_diffusion/ .
```

TODO: record the exact commit hash once confirmed:

```
Commit: <TODO — run: git -C RePaint/ log -1 --format="%H  %ai  %s">
```

Note: `conf_mgt/` and `utils/` in this repo are derived from the same RePaint codebase and are licensed under **CC BY-NC-SA 4.0** (Huawei Technologies Co., Ltd.). See `LICENSE` for implications.

---

### `deepfillv2-pytorch/`

**Used by:** `FID.py` (`states_pt_places2.pth`), `halluc.py` (`states_tf_places2.pth`)

A PyTorch implementation of DeepFill v2. Clone into the repo root:

```bash
git clone <TODO: confirm exact repo URL>
# TODO: record commit hash:
# Commit: <TODO>
```

Two pretrained checkpoints are required under `deepfillv2-pytorch/pretrained/`:

- `states_pt_places2.pth` — PyTorch-native checkpoint (used by `FID.py`)
- `states_tf_places2.pth` — TF-style checkpoint (used by `halluc.py`)

Download links are provided in the deepfillv2-pytorch repository's README.
