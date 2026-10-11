"""
CIFAR-10 Convolutional Neural Network (CNN) in PyTorch
End-to-end training and validation script.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Paths
# ==========================================
DATASET_PATH = r"C:\Users\Rushabh\.cache\kagglehub\competitions\cifar-10"
LABELS_PATH = os.path.join(DATASET_PATH, "trainLabels.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "train", "train")

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. Class Mapping
# ==========================================
CLASS_TO_IDX = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ==========================================
# 3. Custom Dataset
# ==========================================
class Cifar10Dataset(Dataset):
    """Custom PyTorch Dataset for CIFAR-10."""
    def __init__(self, image_ids, labels, img_dir, class_to_idx, transform=None):
        super().__init__()
        self.image_ids = list(image_ids)
        self.labels = list(labels)
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        filename = f"{self.image_ids[index]}.png"
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_name = self.labels[index]
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        return img, label

# ==========================================
# 4. Neural Network Architecture
# ==========================================
class CIFAR10_CNN(nn.Module):
    """Deep CNN for CIFAR-10 with BatchNorm and Dropout."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 (Input: 3 x 32 x 32 -> Output: 32 x 16 x 16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 2 (Input: 32 x 16 x 16 -> Output: 64 x 8 x 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # Block 3 (Input: 64 x 8 x 8 -> Output: 128 x 4 x 4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 5. Main Training Pipeline
# ==========================================
def main():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Could not find labels file at {LABELS_PATH}")

    # Load and split labels
    df = pd.read_csv(LABELS_PATH)
    X = df["id"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # CIFAR-10 Normalization Statistics
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # Transforms (Train has data augmentation, Val has only normalization)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Datasets & Loaders
    train_dataset = Cifar10Dataset(X_train, y_train, TRAIN_IMG_DIR, CLASS_TO_IDX, train_transform)
    val_dataset = Cifar10Dataset(X_test, y_test, TRAIN_IMG_DIR, CLASS_TO_IDX, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model, loss, optimizer, scheduler
    model = CIFAR10_CNN(num_classes=10).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Training History
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    print("\nStarting Training...")
    print("-" * 75)

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        avg_train_loss = total_train_loss / total_train_samples

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / total_val_samples
        val_acc = val_correct / total_val_samples

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch [{epoch + 1:2d}/{EPOCHS:2d}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    print("-" * 75)
    print(f"Final Validation Accuracy: {history['val_acc'][-1] * 100:.2f}%")

    # Save Checkpoint
    checkpoint_path = "cifar10_cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': CLASS_TO_IDX,
        'mean': mean,
        'std': std,
        'final_val_acc': history['val_acc'][-1]
    }, checkpoint_path)
    print(f"Model saved to '{checkpoint_path}'")

if __name__ == "__main__":
    main()
