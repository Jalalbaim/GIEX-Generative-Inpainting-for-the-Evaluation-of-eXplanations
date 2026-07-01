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
