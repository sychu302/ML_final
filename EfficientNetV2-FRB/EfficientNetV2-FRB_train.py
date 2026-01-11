import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 目录配置

DATA_DIR = "/root/autodl-tmp/tobacco/data_old"
SAVE_PATH = "/root/autodl-tmp/tobacco/efficientnetv2_frb_best.pth"

BATCH_SIZE = 32
INPUT_SIZE = 300   
NUM_EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 8

# 创新模块：FRB (Feature Refinement Block)

class FRB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1, groups=channels
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        refined = self.pw(self.dw(x))
        gate = self.gate(x)
        return refined * gate + x


# EfficientNet-V2 + FRB

class EfficientNetV2_FRB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # backbone
        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

        in_channels = self.backbone.classifier[1].in_features

        # 在最后特征层加入FRB
        self.frb = FRB(in_channels)

        # 替换 classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.frb(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x



# 数据增强

def get_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.15)
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return train_tfms, val_tfms



# dataloader（只构建 train / val）

def build_loaders():
    train_tfms, val_tfms = get_transforms()

    image_datasets = {
        'train': datasets.ImageFolder(
            os.path.join(DATA_DIR, "train"), train_tfms
        ),
        'val': datasets.ImageFolder(
            os.path.join(DATA_DIR, "val"), val_tfms
        ),
    }

    loaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == 'train'),
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        for x in ['train', 'val']
    }

    sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    classes = image_datasets['train'].classes
    return loaders, sizes, classes



# 训练

def train_model():
    loaders, sizes, classes = build_loaders()
    num_classes = len(classes)

    model = EfficientNetV2_FRB(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    print("\nStart Training…\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / sizes[phase]
            epoch_loss = running_loss / sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 只用 val 选最优权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, SAVE_PATH)
                print("Saved new best model")

        print()

    print("\nTraining Finished. Best Val Acc =", best_acc)
    return best_wts, classes



# main

if __name__ == "__main__":
    best_wts, classes = train_model()
    print(f"Best weights saved to: {SAVE_PATH}")
