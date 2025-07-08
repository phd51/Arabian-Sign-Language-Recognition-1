import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, cohen_kappa_score,
    log_loss, mean_squared_error
)
import os

# === Step 1: Define Hybrid Model ===
class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridModel, self).__init__()
        self.vgg_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.inception = timm.create_model('inception_v4', pretrained=True, num_classes=0)
        densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        self.densenet_features = densenet.features
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1920 + 1536 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_vgg = self.vgg_features(x)
        x_vgg = nn.AdaptiveAvgPool2d((1, 1))(x_vgg)
        x_vgg = torch.flatten(x_vgg, 1)

        x_incep = self.inception(x)

        x_dense = self.densenet_features(x)
        x_dense = self.densenet_pool(x_dense)
        x_dense = torch.flatten(x_dense, 1)

        x_all = torch.cat((x_vgg, x_incep, x_dense), dim=1)
        return self.classifier(x_all)

# === Step 2: Dataset Loader ===
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️ File not found: {self.image_paths[idx]}")
            img = Image.new("RGB", (299, 299))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# === Step 3: Transformations ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Step 4: Load CSV and Image Paths ===
csv_path = ""
image_folder = ""

df = pd.read_csv(csv_path)
if 'filename' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV must contain 'filename' and 'label' columns.")

df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
image_paths = df['filepath'].tolist()
labels = df['label'].values

# === Step 5: Encode Labels ===
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# === Step 6: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.3, random_state=42)
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# === Step 7: Model Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels_encoded))
model = HybridModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# === Step 8: Evaluation Metrics ===
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        y_true.extend(lbls.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

num_classes = len(np.unique(y_true))
average_type = 'weighted' if num_classes > 2 else 'binary'

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    # Approximate for multi-class
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

# Metrics Calculation
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, average=average_type),
    "Recall (Sensitivity)": recall_score(y_true, y_pred, average=average_type),
    "F1 Score": f1_score(y_true, y_pred, average=average_type),
    "AUC": roc_auc_score(y_true, y_probs, multi_class='ovr') if num_classes > 2 else roc_auc_score(y_true, y_probs[:, 1]),
    "MCC": matthews_corrcoef(y_true, y_pred),
    "Specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
    "FPR": FP / (FP + TN) if (FP + TN) > 0 else 0,
    "FNR": FN / (FN + TP) if (FN + TP) > 0 else 0,
    "G-Mean": np.sqrt(
        (TP / (TP + FN)) * (TN / (TN + FP))
    ) if (TP + FN) > 0 and (TN + FP) > 0 else 0,
    "Kappa": cohen_kappa_score(y_true, y_pred),
    "Log Loss": log_loss(y_true, y_probs),
    "Cross-Entropy Loss": log_loss(y_true, y_probs),
    "MSE": mean_squared_error(y_true, y_pred)
}

print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:25}: {v:.4f}")