import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import timm
from einops import rearrange
import os
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
# === Attention Gate ===
class AttentionGate(nn.Module):
    def __init__(self, in_channels_g, in_channels_l, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(in_channels_g, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(in_channels_l, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

# === RDSC ===
class RDSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

# === Decoder Block ===
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, skip_channels // 2)
        self.rdsc = RDSC(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.rdsc(x)

# === Swin Transformer Backbone ===
class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)

    def forward(self, x):
        return self.model(x)

# === STARDC Segmentation Model ===
class STARDC(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.encoder = SwinBackbone()
        self.decoder4 = DecoderBlock(768, 384, 256)
        self.decoder3 = DecoderBlock(256, 192, 128)
        self.decoder2 = DecoderBlock(128, 96, 64)
        self.decoder1 = DecoderBlock(64, 48, 32)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder4(feats[-1], feats[-2])
        x = self.decoder3(x, feats[-3])
        x = self.decoder2(x, feats[-4])
        x = self.decoder1(x, feats[-5])
        return self.final_conv(x)

# === Dice + Focal Loss ===
class FusionLoss(nn.Module):
    def __init__(self, lambda_factor=1.0, alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.lambda_factor = lambda_factor
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.pow(2).sum() + targets.pow(2).sum() + self.smooth)

        focal_loss = - self.alpha * (1 - preds) ** self.gamma * targets * torch.log(preds + self.smooth) - \
                     (1 - self.alpha) * preds ** self.gamma * (1 - targets) * torch.log(1 - preds + self.smooth)
        focal_loss = focal_loss.mean()

        return dice_loss + self.lambda_factor * focal_loss

# === Segmentation Dataset Loader ===
class SegDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
        self.mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, (mask > 0.5).float()

# === Metrics Function ===
def compute_metrics(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds_np = preds.cpu().numpy().astype(np.uint8)
    targets_np = targets.cpu().numpy().astype(np.uint8)

    batch_iou, batch_dice, batch_acc, batch_bf1 = [], [], [], []

    for i in range(preds_np.shape[0]):
        pred = preds_np[i, 0]
        target = targets_np[i, 0]

        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        iou = intersection / (union + 1e-6)

        dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-6)
        acc = (pred == target).sum() / (pred.size + 1e-6)

        pred_contour = cv2.Canny(pred * 255, 100, 200)
        target_contour = cv2.Canny(target * 255, 100, 200)

        pred_d = distance_transform_edt(1 - pred_contour / 255)
        target_d = distance_transform_edt(1 - target_contour / 255)

        pred_matches = (target_d < 2).astype(np.uint8)
        target_matches = (pred_d < 2).astype(np.uint8)

        tp = (pred_contour / 255 * pred_matches).sum()
        fp = (pred_contour / 255).sum() - tp
        fn = (target_contour / 255).sum() - tp
        bf1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)

        batch_iou.append(iou)
        batch_dice.append(dice)
        batch_acc.append(acc)
        batch_bf1.append(bf1)

    return {
        "IoU": np.mean(batch_iou),
        "DSC (Dice)": np.mean(batch_dice),
        "Pixel Accuracy": np.mean(batch_acc),
        "Boundary F1": np.mean(batch_bf1)
    }
input_folder = ''
output_folder = ''
os.makedirs(output_folder, exist_ok=True)

valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = Image.open(input_path).convert("RGBA")
        output = remove(image)

        # ✅ Fix: convert RGBA to RGB if saving as JPG
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            output = output.convert("RGB")
        output.save(output_path)

        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Original')
        axs[0].axis('off')

        axs[1].imshow(output)
        axs[1].set_title('Background Removed')
        axs[1].axis('off')

        plt.suptitle(f"Processed: {filename}")
        plt.tight_layout()
        plt.show()
