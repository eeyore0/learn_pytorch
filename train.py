# train.py (版本 10.1 - 完整且修正的版本)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau

from unet_model import UNet

# --- 1. 定义超参数和配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
BATCH_SIZE = 6
NUM_EPOCHS = 30
PATCH_SIZE = 256
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
NEGATIVE_SAMPLE_RATIO = 0.5

IMAGE_TYPE_TO_TRAIN = 'CM'

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


# --- 2. 创建自定义数据集类 (已补全) ---
class CDD_CESM_Patch_Dataset(Dataset):
    def __init__(self, annotations_df: pd.DataFrame, full_annotations_df: pd.DataFrame, img_dir: str, patch_size: int,
                 augmentations=None):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.full_annotations = full_annotations_df
        self.patches = self._create_patches(annotations_df)

    def _create_patches(self, df):
        patch_list = []
        desc = f"Creating Positive Patches for {len(df)} annotations"
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=desc):
            img_name = row['#filename']
            try:
                region = json.loads(row['region_shape_attributes'])
                center_x = int(np.mean(region['all_points_x']))
                center_y = int(np.mean(region['all_points_y']))
                patch_list.append({'name': img_name, 'center_x': center_x, 'center_y': center_y})
            except (json.JSONDecodeError, KeyError):
                continue

        num_positive_patches = len(patch_list)
        print(f"创建了 {num_positive_patches} 个正样本图块。")

        num_negative_to_add = int(num_positive_patches * NEGATIVE_SAMPLE_RATIO)
        all_image_files = df['#filename'].unique()

        image_masks = {}
        for img_name in all_image_files:
            same_img_annotations = self.full_annotations[self.full_annotations['#filename'] == img_name]
            temp_img_path = os.path.join(self.img_dir, img_name)
            temp_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
            if temp_img is None: continue

            full_mask = np.zeros(temp_img.shape, dtype=np.uint8)
            for _, row in same_img_annotations.iterrows():
                try:
                    region = json.loads(row['region_shape_attributes'])
                    points = np.array(list(zip(region['all_points_x'], region['all_points_y'])), dtype=np.int32)
                    cv2.fillPoly(full_mask, [points], 255)
                except:
                    continue
            image_masks[img_name] = full_mask

        negative_patches_added = 0
        desc_neg = f"Creating {num_negative_to_add} Negative Patches"
        with tqdm(total=num_negative_to_add, desc=desc_neg) as pbar:
            while negative_patches_added < num_negative_to_add:
                img_name = random.choice(all_image_files)
                if img_name not in image_masks: continue

                full_mask = image_masks[img_name]
                h, w = full_mask.shape

                rand_center_x = random.randint(self.patch_size // 2, w - self.patch_size // 2)
                rand_center_y = random.randint(self.patch_size // 2, h - self.patch_size // 2)

                half_patch = self.patch_size // 2
                y1, y2 = rand_center_y - half_patch, rand_center_y + half_patch
                x1, x2 = rand_center_x - half_patch, rand_center_x + half_patch

                if np.sum(full_mask[y1:y2, x1:x2]) == 0:
                    patch_list.append({'name': img_name, 'center_x': rand_center_x, 'center_y': rand_center_y})
                    negative_patches_added += 1
                    pbar.update(1)

        print(f"添加了 {negative_patches_added} 个负样本图块。总图块数: {len(patch_list)}")
        random.shuffle(patch_list)
        return patch_list

    # +++ 补全：__len__ 方法 +++
    def __len__(self):
        return len(self.patches)

    # +++ 补全：__getitem__ 方法 +++
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


def main():
    train_augs = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_augs = A.Compose([
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

    train_dataset = CDD_CESM_Patch_Dataset(train_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, train_augs)
    val_dataset = CDD_CESM_Patch_Dataset(val_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, val_augs)
    test_dataset = CDD_CESM_Patch_Dataset(test_df, full_df, IMAGE_ROOT_DIR, PATCH_SIZE, val_augs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)  # 增加patience

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