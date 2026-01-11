import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 基本配置

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_DIR = "/root/autodl-tmp/tobacco/data_old"
MODEL_PATH = "/root/autodl-tmp/tobacco/efficientnetv2_frb_best.pth"
CM_SAVE_PATH = "/root/autodl-tmp/tobacco/efficientnetv2_frb_cm_test.png"

BATCH_SIZE = 32
INPUT_SIZE = 300
NUM_WORKERS = 8


# FRB 模块（和训练脚本完全一致）

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

        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        in_channels = self.backbone.classifier[1].in_features

        self.frb = FRB(in_channels)

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



# 1) 构建 test DataLoader

def get_test_loader():
    test_tf = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=test_tf
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return loader, test_dataset.classes



# 2) 在测试集上评估

def evaluate_on_test():
    test_loader, classes = get_test_loader()
    num_classes = len(classes)

    # 建模
    model = EfficientNetV2_FRB(num_classes=num_classes).to(DEVICE)

    # 加载 best 权重
    print(f"Loading model weights from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 整体准确率
    acc = 100.0 * correct / total
    print(f"Test Accuracy (EffNetV2-FRB): {acc:.2f}%\n")

    # 分类报告
    print("Classification report:")
    print(classification_report(all_labels, all_preds,
                                target_names=classes, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=classes, yticklabels=classes)
    plt.title("EfficientNetV2-FRB Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH)
    print(f"Confusion matrix saved to: {CM_SAVE_PATH}")



# main

if __name__ == "__main__":
    evaluate_on_test()
