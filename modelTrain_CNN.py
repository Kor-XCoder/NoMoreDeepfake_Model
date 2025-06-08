import albumentations as AL
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from PIL import Image
import glob, os
import numpy as np
import torch, torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

train_tf = AL.Compose([
    AL.RandomResizedCrop((224,224), (0.9, 1.0)),
    AL.HorizontalFlip(p=0.5),
    AL.OneOf([
        AL.MotionBlur(p=0.2),
        AL.GaussianBlur(p=0.2),
    ], p=0.2),
    AL.ImageCompression(
        p=0.3,
    ),
    AL.ColorJitter(p=0.3),
    AL.Normalize(),
    ToTensorV2(),
])

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob.glob(os.path.join(root_dir, '*/*.jpg'))
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        label = 1 if 'Fake' in path else 0
        img = Image.open(path).convert('RGB')
        return self.transform(image=np.array(img))['image'], label

class EfficientNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_ch = self.backbone.classifier[1].in_features
        # 기존 분류기 제거 후 2-클래스 FC 추가
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4, inplace=True),
            nn.Linear(in_ch, 2)
        )
    def forward(self, x):
        return self.backbone(x)


def main():
    # ─── 하이퍼파라미터 ───
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 30
    DEVICE = "cpu"
    if torch.backends.mps.is_available():        # M1/M2 + Metal 지원 여부
        DEVICE = "mps"
    elif torch.cuda.is_available():              # (예: 외장 NVIDIA eGPU)
        DEVICE = "cuda"
    print(f"Using device: {DEVICE}")

    # ─── 데이터로더 ───
    train_set = DeepfakeDataset('/root/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset/Train', transform=train_tf)
    val_set = DeepfakeDataset('/root/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset/Validation', transform=train_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=False)

    # ─── 모델·손실·옵티마이저 ───
    model = EfficientNetDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0.0
    for epoch in range(EPOCHS):
        # ── 1) TRAIN ──────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] Train", leave=False) as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # ── 2) VALIDATION ────────────────────────────────────────
        model.eval()
        val_loss, y_true, y_score = 0.0, [], []
        with torch.no_grad(), tqdm(val_loader, desc=f"[Epoch {epoch+1}] Val ", leave=False) as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)

                probs = F.softmax(logits, dim=1)[:, 1]          # 딥페이크 클래스 확률
                val_loss += loss.item() * imgs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_score.extend(probs.cpu().numpy())

        epoch_train_loss = running_loss / len(train_set)
        epoch_val_loss = val_loss / len(val_set)
        epoch_auc = roc_auc_score(y_true, y_score)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss {epoch_train_loss:.4f} | "
              f"Val Loss {epoch_val_loss:.4f} | "
              f"AUC {epoch_auc:.4f}")

        # ── 3) 모델 저장 ─────────────────────────────────────────
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            torch.save(model.state_dict(), "best_detector.pth")

if __name__ == "__main__":
    main()
