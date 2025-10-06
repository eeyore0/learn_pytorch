# train.py (版本 9 - 实施分块训练)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau

from unet_model import UNet

# --- 1. 定义超参数和配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 6 # 分块后数据量大增，可以适当增加Batch Size
NUM_EPOCHS = 100 # 需要更多的轮数来学习
PATCH_SIZE = 256 # 定义图块大小
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
IMAGE_TYPE_TO_TRAIN = 'DM'

ANNOTATIONS_CSV_PATH = "PKG-CDD-CESM/Radiology_hand_drawn_segmentations_v2.csv"
if IMAGE_TYPE_TO_TRAIN == 'DM':
    IMAGE_ROOT_DIR = "PKG-CDD-CESM/CDD-CESM/Low energy images of CDD-CESM/"
else:
    IMAGE_ROOT_DIR = "PKG-CDD-CESM/CDD-CESM/Subtracted images of CDD-CESM/"


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds, targets = preds.contiguous().view(-1), targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5,
                 smooth=1e-6): super().__init__(); self.weight = weight; self.bce = nn.BCEWithLogitsLoss(); self.dice = DiceLoss(
        smooth)

    def forward(self, preds, targets): return self.weight * self.bce(preds, targets) + (1 - self.weight) * self.dice(
        preds, targets)


def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds, targets = preds.contiguous().view(-1), targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# --- 2. 创建自定义数据集类 (完全重构以实现分块) ---
# --- 2. 创建自定义数据集类 (已修正) ---
class CDD_CESM_Patch_Dataset(Dataset):
    # +++ 修正：__init__ 接收 full_annotations_df +++
    def __init__(self, annotations_df: pd.DataFrame, full_annotations_df: pd.DataFrame, img_dir: str, patch_size: int,
                 augmentations=None):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.full_annotations = full_annotations_df  # 存储完整的标注信息
        self.patches = self._create_patches(annotations_df)

    def _create_patches(self, df):
        patch_list = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Creating Patches for {len(df)} annotations"):
            img_name = row['#filename']
            try:
                region = json.loads(row['region_shape_attributes'])
                center_x = int(np.mean(region['all_points_x']))
                center_y = int(np.mean(region['all_points_y']))
                patch_list.append({'name': img_name, 'center_x': center_x, 'center_y': center_y})
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"创建了 {len(patch_list)} 个正样本图块。")
        return patch_list

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        img_name = patch_info['name']
        center_x = patch_info['center_x']
        center_y = patch_info['center_y']

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None: return torch.zeros((1, self.patch_size, self.patch_size)), torch.zeros(
            (1, self.patch_size, self.patch_size))

        h, w = image.shape
        half_patch = self.patch_size // 2

        x1, y1 = max(0, center_x - half_patch), max(0, center_y - half_patch)
        x2, y2 = min(w, x1 + self.patch_size), min(h, y1 + self.patch_size)
        x1, y1 = max(0, x2 - self.patch_size), max(0, y2 - self.patch_size)

        image_patch = image[y1:y2, x1:x2]

        same_img_annotations = self.full_annotations[self.full_annotations['#filename'] == img_name]
        mask = np.zeros(image.shape, dtype=np.uint8)
        for _, row in same_img_annotations.iterrows():
            try:
                region = json.loads(row['region_shape_attributes'])
                points = np.array(list(zip(region['all_points_x'], region['all_points_y'])), dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            except:
                continue

        mask_patch = mask[y1:y2, x1:x2]

        if self.augmentations:
            augmented = self.augmentations(image=image_patch, mask=mask_patch)
            image = augmented['image']
            mask = augmented['mask']

        # +++ THIS IS THE FIX +++
        # The error indicates that the 'image' tensor is a ByteTensor (uint8).
        # We must explicitly convert it to a FloatTensor before returning.
        # Note: ToTensorV2 should have already scaled values to [0,1],
        # so we only need to change the data type.
        image = image.float()

        mask = mask.float()
        mask[mask > 0] = 1.0

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    running_loss, running_dice = 0.0, 0.0
    for data, targets in loop:
        if data.nelement() == 0: continue
        data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        dice_score = dice_coefficient(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_dice += dice_score.item()
        loop.set_postfix(loss=loss.item(), dice=dice_score.item())
    avg_loss = running_loss / len(loader) if loader else 0
    avg_dice = running_dice / len(loader) if loader else 0
    return avg_loss, avg_dice


def evaluate_fn(loader, model, loss_fn, desc="Validation"):
    model.eval()
    loop = tqdm(loader, leave=True, desc=desc)
    running_loss, running_dice = 0.0, 0.0
    with torch.no_grad():
        for data, targets in loop:
            if data.nelement() == 0: continue
            data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            dice_score = dice_coefficient(predictions, targets)
            running_loss += loss.item()
            running_dice += dice_score.item()
            loop.set_postfix(loss=loss.item(), dice=dice_score.item())
    avg_loss = running_loss / len(loader) if loader else 0
    avg_dice = running_dice / len(loader) if loader else 0
    return avg_loss, avg_dice


# --- 4. 主执行函数 (已更新) ---
def main():
    train_augs = A.Compose([
        # A.Resize(PATCH_SIZE, PATCH_SIZE) # 不再需要Resize，因为我们直接提取该尺寸的块
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_augs = A.Compose([
        # A.Resize(PATCH_SIZE, PATCH_SIZE)
        ToTensorV2(),
    ])

    print(f"设备: {DEVICE}")
    print(f"训练图像类型: {'Low energy (DM)' if IMAGE_TYPE_TO_TRAIN == 'DM' else 'Subtracted (CM)'}")

    full_df = pd.read_csv(ANNOTATIONS_CSV_PATH)
    full_df = full_df[full_df['region_shape_attributes'] != '{}']
    filter_key = f"_{IMAGE_TYPE_TO_TRAIN}_"
    filtered_df = full_df[full_df['#filename'].str.contains(filter_key)].copy()

    train_df, temp_df = train_test_split(filtered_df, test_size=(TEST_SPLIT + VAL_SPLIT), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT)), random_state=42)
    print(f"总标注数: {len(filtered_df)} | 训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

    # +++ 修正：将 full_df 传递给Dataset +++
    train_dataset = CDD_CESM_Patch_Dataset(train_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, train_augs)
    val_dataset = CDD_CESM_Patch_Dataset(val_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, val_augs)
    test_dataset = CDD_CESM_Patch_Dataset(test_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, val_augs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_val_dice = -1.0
    model_save_path = f"best_model_{IMAGE_TYPE_TO_TRAIN}.pth"

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss, train_dice = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, val_dice = evaluate_fn(val_loader, model, loss_fn, desc="Validation")

        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Dice: {val_dice:.4f}")

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"  ---> 新的最佳模型已保存到 {model_save_path} (Validation Dice: {best_val_dice:.4f})")

    print("\n--- 训练完成，开始在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_dice = evaluate_fn(test_loader, model, loss_fn, desc="Testing")
    print(f"\n最终测试集性能:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()