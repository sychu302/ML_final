import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 配置 
DATA_DIR = "/root/autodl-tmp/tobacco/data_old"
MODEL_PATH = "/root/autodl-tmp/tobacco/baseline_resnet50_pure.pth"
CM_SAVE_PATH = "/root/autodl-tmp/tobacco/resnet50_test_cm.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
INPUT_SIZE = 224

# 准备测试数据 
test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("正在加载测试集...")
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_tfms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
class_names = test_dataset.classes
num_classes = len(class_names)

# 构建标准 ResNet50 骨架 
print("构建模型骨架...")
model = models.resnet50(weights=None) # 不下载权重，因为我们要加载自己的
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

# 加载权重 
print(f"正在加载权重: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("权重加载成功！")
else:
    print(f"错误：找不到权重文件 {MODEL_PATH}")
    print("请先运行上面的训练脚本！")
    exit()

# 执行测试
print("\n========== Test Set Evaluation ==========")
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
print(f"ResNet50 (Pure) Test Accuracy: {acc*100:.2f}%")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# 绘制测试集混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"ResNet50 Pure - Test Set Confusion Matrix\nAcc: {acc*100:.2f}%")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(CM_SAVE_PATH)
print(f"测试集混淆矩阵已保存: {CM_SAVE_PATH}")
