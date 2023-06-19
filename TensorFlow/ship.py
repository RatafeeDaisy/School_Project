import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import confusion_matrix, roc_curve,\
    precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(0)


# 定义数据集类
class ShipDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_list = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = int(image_name[0])
        return image, label


# 定义船舶检测模型
class ShipDetectionModel(nn.Module):
    def __init__(self):
        super(ShipDetectionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        return output


# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)  # 将模型移动到设备上
    best_val_acc = 0.0
    best_model_weights = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # 在验证集上评估模型
        val_acc = evaluate_model(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)
    return model


# 定义模型评估函数
def evaluate_model(model, data_loader, device):
    model = model.to(device)  # 将模型移动到设备上
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


# 定义模型预测函数
def predict(model, image, device):
    image = image.unsqueeze(0)
    image = image.to(device)  # 将输入数据移动到设备上
    model = model.to(device)  # 将模型移动到设备上
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = predicted.item()
    return predicted_label


# 设置数据目录和超参数
data_dir = "D:/School/School_Project/TensorFlow/archive/shipsnet/shipsnet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = ShipDataset(data_dir, transform=transform)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型、损失函数和优化器
model = ShipDetectionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# 保存模型
torch.save(model.state_dict(), "best_ship_detection_model.pt")

# 加载模型
loaded_model = ShipDetectionModel()
loaded_model.load_state_dict(torch.load("best_ship_detection_model.pt"))
loaded_model = loaded_model.to(device)  # 将加载的模型移动到设备上

# 生成预测结果和真实标签
predictions = []
labels = []
with torch.no_grad():
    for images, target_labels in val_loader:
        images = images.to(device)
        target_labels = target_labels.to(device)
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        labels.extend(target_labels.tolist())

# 计算混淆矩阵
cm = confusion_matrix(labels, predictions)
classes = ["非船", "船"]

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("混淆矩阵", fontsize=16)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.show()

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(labels, predictions)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率', fontsize=12)
plt.ylabel('真阳性率', fontsize=12)
plt.title('ROC 曲线', fontsize=16)
plt.legend(loc="lower right")
plt.show()

# 计算精确率和召回率
precision, recall, _ = precision_recall_curve(labels, predictions)

# 计算平均精确率
avg_precision = np.average(precision)

# 绘制精确率-召回率曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'平均精确率 = {avg_precision:.2f}')
plt.xlabel('召回率', fontsize=12)
plt.ylabel('精确率', fontsize=12)
plt.title('精确率-召回率曲线', fontsize=16)
plt.legend(loc="lower left")
plt.show()

# 打印保存的最佳模型的准确率和评估分数
accuracy = evaluate_model(loaded_model, val_loader, device)
print(f"最佳模型准确率: {accuracy * 100:.2f}%")
print(f"AUC值: {roc_auc:.2f}")
print(f"平均精确率: {avg_precision:.2f}")
