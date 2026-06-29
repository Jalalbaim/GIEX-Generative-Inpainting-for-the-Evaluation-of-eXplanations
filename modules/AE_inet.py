"""
This script will introduce an AE trained on imagenet, that helps to preview indistributionness
@author: J.BAIM
"""

import torch
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

BATCH_SIZE=32
PATH = "data/weights/resnet50-11ad3fa6.pth"

def load_model(path):
    model = models.resnet50(weights=None)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_data_loaders_and_classes(path, batch_size=BATCH_SIZE):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    full_ds = datasets.ImageFolder(root=path, transform=tf)
    bg_label = "BACKGROUND_Google"
    orig_idx_to_class = {v:k for k,v in full_ds.class_to_idx.items()}
    if bg_label in full_ds.class_to_idx:
        bg_idx = full_ds.class_to_idx[bg_label]
        filtered_samples = [(p,idx) for p,idx in full_ds.samples if idx != bg_idx]
        classes = [c for c in full_ds.classes if c != bg_label]
        new_class_to_idx = {c:i for i,c in enumerate(classes)}
        remapped = [
            (p, new_class_to_idx[orig_idx_to_class[idx]])
            for p,idx in filtered_samples
        ]
        full_ds.samples     = remapped
        full_ds.targets     = [t for _,t in remapped]
        full_ds.classes     = classes
        full_ds.class_to_idx= new_class_to_idx
    else:
        classes = full_ds.classes

    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    n_test =  n_val - n_val//2
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_dtst, test_ds = random_split(val_ds, [n_val//2, n_test])
    

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dtst,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Train / Val / Test sizes: {len(train_ds)} / {len(val_dtst)} / {len(test_ds)}")
    print(f"Number of classes (after excluding BG): {len(classes)}")
    return train_loader, val_loader, test_loader,classes


class ResNetAutoEncoder(nn.Module):
    def __init__(self, encoder, num_classes=1000):
        super(ResNetAutoEncoder, self).__init__()
        # Use pretrained ResNet-50 as encoder
        self.encoder = encoder
        # Build only the decoder part
        self.decoder = ResNetDecoder()
        
    def forward(self, x):
        # Extract features manually instead of using a dedicated method
        # Standard ResNet forward path until the final layer
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        features = self.encoder.layer4(x)  # Features before avgpool and fc
        
        # Pass features through decoder
        reconstructed = self.decoder(features)
        return reconstructed

class ResNetDecoder(nn.Module):
    def __init__(self):
        super(ResNetDecoder, self).__init__()
        
        configs = [3, 6, 4, 3]
        
        self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
        self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
        self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
        self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])
  
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )
        
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)
        
        return x

class DecoderBottleneckBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()
        
        for i in range(layers):
            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
                        
            self.add_module('%02d DecoderLayer' % i, layer)
             
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()
        
        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
        
    def forward(self, x):
        identity = x
        
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)
        
        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)
        
        x = x + identity
        
        return x

# def load_pretrained_model(model_path, weights_only=False):
#     """
#     Load your pretrained ResNet-50 model
#     """
#     # Use weights_only=True to avoid security warnings and potential issues
#     encoder = torch.load(model_path, weights_only=weights_only)
#     encoder.eval()
#     return encoder

def get_AE(model_path,device):
    """
    Load the autoencoder model
    """
    PATH = "data/weights/resnet50-11ad3fa6.pth"
    encoder = load_model(PATH)
    encoder.fc = nn.Identity()
    AE = ResNetAutoEncoder(encoder, num_classes=1000).to(device)
    state = torch.load(model_path, map_location=device)
    AE.load_state_dict(state)
    AE.eval()
    return AE


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device : ", device)

    data_path = "/kaggle/input/imagenetmini-1000/imagenet-mini/train"
   

    encoder = load_model(PATH)
    
    # Remove fc layer
    encoder.fc = nn.Identity()
    print("encoder loaded successfully")
    
    autoencoder = ResNetAutoEncoder(encoder, num_classes=1000).to(device)
    print("AE built successfully")
    
    # data
    print("data....")
    train_loader, val_loader, test_loader,classes = get_data_loaders_and_classes(data_path)
    print(f"Train / Val sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    print(f"Number of classes (after excluding BG): {len(classes)}")

    # retrain the decoder
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 10

    print("Starting training...")
    batch_size = 32
    # Training loop with validation by batch
    scaler = GradScaler()

    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader, 1):
            images = images.to(device)

            optimizer.zero_grad()
            # --------------------
            # 1) Forward + loss
            # --------------------
            with autocast():
                outputs = autoencoder(images)
                loss = criterion(outputs, images)

            # --------------------
            # 2) Backward + step
            # --------------------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # --------------------
            # 3) Log per-batch
            # --------------------
            if batch_idx % 10 == 0:
                avg = running_loss / 10
                print(f"Epoch {epoch+1}/{num_epochs}  "
                    f"Batch {batch_idx}/{len(train_loader)}  "
                    f"Loss {avg:.4f}")
                running_loss = 0.0

        # ------------------
        # 4) Validation pass
        # ------------------
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon = autoencoder(images)
                val_loss += criterion(recon, images).item()

        val_loss /= len(val_loader)
        print(f"--- Epoch {epoch+1} validation loss: {val_loss:.4f} ---\n")
        torch.cuda.empty_cache()


    # test the autoencoder
    autoencoder.eval()
    img_test, _ = next(iter(test_loader))
    with torch.no_grad():
        reconstructed = autoencoder(img_test.to(device))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(img_test[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    torch.save(autoencoder, './AE_imagenet_.pth')