# swim-t

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 配置参数
DATA_DIR = "/root/autodl-tmp/tobacco/data_old"
MODEL_PATH = "/root/autodl-tmp/tobacco/baseline_swin_t.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 8
INPUT_SIZE = 224

# 数据准备
test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("正在加载测试集...")
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_tfms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
class_names = test_dataset.classes
num_classes = len(class_names)
print(f"类别: {class_names}")

# 构建 Swin-T 模型
print("正在构建 Swin Transformer Tiny...")
model = models.swin_t(weights=None)
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, num_classes)
model.to(DEVICE)

# 加载训练好的权重
print(f"正在加载训练权重: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # 兼容 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=True)
        print("权重加载成功 (Strict Mode)！")
        
    except Exception as e:
        print(f"权重加载失败: {e}")
        print("尝试非严格加载...")
        model.load_state_dict(new_state_dict, strict=False)
else:
    print(f"错误: 找不到权重文件 {MODEL_PATH}")
    print("请先运行训练脚本生成权重！")
    exit()

# ]开始评估
print("开始 Swin-T 测试集评估")
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

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

acc = correct / total
print(f"Swin-T 最终准确率: {acc*100:.2f}%")

# 输出报告与绘图
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Swin-T Baseline Confusion Matrix")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()

save_path = "/root/autodl-tmp/tobacco/swin_t_final_cm.png"
plt.savefig(save_path)
print(f"混淆矩阵已保存至: {save_path}")
