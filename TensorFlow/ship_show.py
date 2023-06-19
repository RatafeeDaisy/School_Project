import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 定义数据集类
class ShipDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, image_path

# 定义船舶检测模型
class ShipDetectionModel(nn.Module):
    def __init__(self):
        super(ShipDetectionModel, self).__init__()
        # 定义你的模型结构，保持与训练时的模型结构一致

    def forward(self, x):
        # 定义模型的前向传播逻辑
        pass

# 加载保存的最佳模型
model = ShipDetectionModel()
model.load_state_dict(torch.load("best_ship_detection_model.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 指定待识别图片的路径
image_dir = "D:/School/School_Project/TensorFlow/archive/scenes/scenes"

# 获取待识别图片的路径列表
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

# 创建数据集
dataset = ShipDataset(image_paths, transform=transform)

# 创建数据加载器
batch_size = 1  # 每次处理一张图片
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# 对每张图片进行船舶识别
for images, image_paths in data_loader:
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted_label = predicted.item()

    # 获取图片路径
    image_path = image_paths[0]

    # 加载原始图片
    original_image = cv2.imread(image_path)

    # 在识别出的船舶上画框标记
    if predicted_label == 1:  # 如果识别结果为船舶
        # 获取框的坐标（示例为随机生成的坐标）
        x1, y1, x2, y2 = 100, 100, 200, 200

        # 在原始图片上画框
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示原始图片和标记后的图片
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
