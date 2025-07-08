--------------------------comparison Method-------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# === Step 1: Define LBP Layer and LBPNet Model ===
class LBPLayer(nn.Module):
    def __init__(self):
        super(LBPLayer, self).__init__()

    def forward(self, x):
        B, C, H, W = x.size()
        out = torch.zeros_like(x)

        for c in range(C):
            center = x[:, c:c+1, :, :]
            lbp = torch.zeros_like(center)

            for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1), (1, 0),  (1, 1)]:
                shifted = F.pad(center, (1, 1, 1, 1), mode='reflect')
                shifted = shifted[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
                lbp += (shifted >= center).float()

            out[:, c, :, :] = lbp[:, 0, :, :] / 8.0
        return out

class LBPNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LBPNet, self).__init__()
        self.lbp = LBPLayer()
        self.bn1 = nn.BatchNorm2d(1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.lbp(x)
        x = self.bn1(x)
        x = F.relu(self.pool1(x))  # output: (B, 1, 14, 14)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
            img = Image.open(self.image_paths[idx]).convert("L")  # Grayscale
        except FileNotFoundError:
            print(f"⚠️ File not found: {self.image_paths[idx]}")
            img = Image.new("L", (28, 28))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# === Step 3: Transformations (Grayscale & Resize to 28x28) ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
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
model = LBPNet(num_classes=num_classes).to(device)

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

cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

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
-------------------------------------------------------------------------
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, cohen_kappa_score,
    log_loss, mean_squared_error
)
from tqdm import tqdm

# === Step 1: Dataset Loader for DFDT ===
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️ File not found: {self.image_paths[idx]}")
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img

# === Step 2: Load Data ===
csv_path = ""
image_folder = ""

df = pd.read_csv(csv_path)
if 'filename' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV must contain 'filename' and 'label' columns.")

df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
image_paths = df['filepath'].tolist()
labels = df['label'].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.3, random_state=42)

# === Step 3: Feature Extraction using ResNet18 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(image_paths):
    dataset = ImageDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    features = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            out = resnet(imgs).squeeze()
            if len(out.shape) == 1:
                out = out.unsqueeze(0)
            features.append(out.cpu().numpy())
    return np.vstack(features)

X_train = extract_features(X_train_paths)
X_test = extract_features(X_test_paths)

# === Step 4: Train Decision Tree ===
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# === Step 5: Predict and Evaluate ===
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)

num_classes = len(np.unique(y_test))
average_type = 'weighted' if num_classes > 2 else 'binary'

cm = confusion_matrix(y_test, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average=average_type),
    "Recall (Sensitivity)": recall_score(y_test, y_pred, average=average_type),
    "F1 Score": f1_score(y_test, y_pred, average=average_type),
    "AUC": roc_auc_score(y_test, y_probs, multi_class='ovr') if num_classes > 2 else roc_auc_score(y_test, y_probs[:, 1]),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "Specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
    "FPR": FP / (FP + TN) if (FP + TN) > 0 else 0,
    "FNR": FN / (FN + TP) if (FN + TP) > 0 else 0,
    "G-Mean": np.sqrt(
        (TP / (TP + FN)) * (TN / (TN + FP))
    ) if (TP + FN) > 0 and (TN + FP) > 0 else 0,
    "Kappa": cohen_kappa_score(y_test, y_pred),
    "Log Loss": log_loss(y_test, y_probs),
    "Cross-Entropy Loss": log_loss(y_test, y_probs),
    "MSE": mean_squared_error(y_test, y_pred)
}

print("\n=== DFDT Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:25}: {v:.4f}")
------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
from tqdm import tqdm

# === CNN-LSTM Model ===
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 512, 7, 7)

        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Ensure fixed size
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)             # (B, 512, 7, 7)
        x = self.pool(x)            # (B, 512, 7, 7)
        x = x.permute(0, 2, 3, 1)   # (B, 7, 7, 512)
        B, T1, T2, F = x.shape
        x = x.reshape(B, T1 * T2, F)  # (B, 49, 512)
        x, _ = self.lstm(x)          # (B, 49, 128)
        x = x[:, -1, :]              # (B, 128)
        x = self.fc(x)               # (B, num_classes)
        return x

# === Dataset Loader ===
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
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Load Data ===
csv_path = ""
image_folder = ""

df = pd.read_csv(csv_path)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
image_paths = df['filepath'].tolist()
labels = df['label'].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.3, random_state=42)
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# === Model Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels_encoded))
model = CNN_LSTM(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Evaluation ===
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

# === Metrics ===
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

num_classes = len(np.unique(y_true))
average_type = 'weighted' if num_classes > 2 else 'binary'

cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

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
    "G-Mean": np.sqrt((TP / (TP + FN)) * (TN / (TN + FP))) if (TP + FN) > 0 and (TN + FP) > 0 else 0,
    "Kappa": cohen_kappa_score(y_true, y_pred),
    "Log Loss": log_loss(y_true, y_probs),
    "Cross-Entropy Loss": log_loss(y_true, y_probs),
    "MSE": mean_squared_error(y_true, y_pred)
}

print("\n=== CNN-LSTM Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:25}: {v:.4f}")
-------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
from tqdm import tqdm

# === VGG-19 Model ===
class VGG19Classifier(nn.Module):
    def __init__(self, num_classes):
        super(VGG19Classifier, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = vgg.features  # Convolutional part
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # (B, 512, 7, 7)
        x = self.avgpool(x)           # (B, 512, 7, 7)
        x = torch.flatten(x, 1)       # (B, 512*7*7)
        x = self.classifier(x)        # (B, num_classes)
        return x

# === Dataset Loader ===
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
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Load Data ===
csv_path = ""
image_folder = ""

df = pd.read_csv(csv_path)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
image_paths = df['filepath'].tolist()
labels = df['label'].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.3, random_state=42)
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# === Model Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels_encoded))
model = VGG19Classifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Evaluation ===
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

# === Metrics ===
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

num_classes = len(np.unique(y_true))
average_type = 'weighted' if num_classes > 2 else 'binary'

cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

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
    "G-Mean": np.sqrt((TP / (TP + FN)) * (TN / (TN + FP))) if (TP + FN) > 0 and (TN + FP) > 0 else 0,
    "Kappa": cohen_kappa_score(y_true, y_pred),
    "Log Loss": log_loss(y_true, y_probs),
    "Cross-Entropy Loss": log_loss(y_true, y_probs),
    "MSE": mean_squared_error(y_true, y_pred)
}

print("\n=== VGG-19 Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:25}: {v:.4f}")
-----------------------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
from tqdm import tqdm
import pretrainedmodels

# === Inception v4 Model ===
class InceptionV4Classifier(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV4Classifier, self).__init__()
        self.model = pretrainedmodels.__dict__['inceptionv4'](pretrained='imagenet')
        in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# === Dataset Loader ===
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

# === Transformations for Inception v4 (expects 299x299) ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Load Data ===
csv_path = ""
image_folder = ""

df = pd.read_csv(csv_path)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
image_paths = df['filepath'].tolist()
labels = df['label'].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.3, random_state=42)
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels_encoded))
model = InceptionV4Classifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training ===
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Evaluation ===
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

# === Metrics ===
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

num_classes = len(np.unique(y_true))
average_type = 'weighted' if num_classes > 2 else 'binary'

cm = confusion_matrix(y_true, y_pred)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
else:
    TN = np.sum([cm[i, i] for i in range(num_classes)])
    FP = np.sum(cm) - TN
    FN = FP
    TP = TN

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
    "G-Mean": np.sqrt((TP / (TP + FN)) * (TN / (TN + FP))) if (TP + FN) > 0 and (TN + FP) > 0 else 0,
    "Kappa": cohen_kappa_score(y_true, y_pred),
    "Log Loss": log_loss(y_true, y_probs),
    "Cross-Entropy Loss": log_loss(y_true, y_probs),
    "MSE": mean_squared_error(y_true, y_pred)
}

print("\n=== Inception-v4 Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:25}: {v:.4f}")
