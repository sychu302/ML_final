import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 配置

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = "/root/autodl-tmp/tobacco/data_old"
MODEL_PATH = "/root/autodl-tmp/tobacco/efficientnetv2_baseline_best.pth"
CM_SAVE_PATH = "/root/autodl-tmp/tobacco/efficientnetv2_baseline_cm_test.png"

BATCH_SIZE = 32
INPUT_SIZE = 300 
NUM_WORKERS = 8

# 模型定义

class EfficientNetV2_Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)

        in_channels = self.backbone.classifier[1].in_features

    
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# 3) 数据加载 (Test Set)

def get_test_loader():
   
    test_tfms = transforms.Compose([
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
        transform=test_tfms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes

# 评估流程

def evaluate():
    # 准备数据
    test_loader, classes = get_test_loader()
    num_classes = len(classes)
    print(f"Test Classes: {classes}")

    # 初始化模型
    print("Building EfficientNetV2 Baseline...")
    model = EfficientNetV2_Baseline(num_classes).to(DEVICE)

    # 加载权重
    print(f"Loading weights from: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"File not found: {MODEL_PATH}")
        print("请先运行训练脚本！")
        return

    # 开始推理
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print("\nStarting Inference on Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 输出结果
    acc = 100.0 * correct / total
    print(f"Test Accuracy (EfficientNetV2 Baseline): {acc:.2f}%")

    # 分类报告
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f"EfficientNetV2 Baseline - Test CM\nAcc: {acc:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH)
    print(f"Confusion Matrix saved to: {CM_SAVE_PATH}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    evaluate()
