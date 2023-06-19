import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 设置狗和猫图像文件夹路径
dog_folder = 'D:/School/School_Project/Bananice/Dataset/PetImages/Dog'
cat_folder = 'D:/School/School_Project/Bananice/Dataset/PetImages/Cat'

# 读取狗图像文件并创建标签
dog_images = []
dog_labels = []
for filename in tqdm(os.listdir(dog_folder), desc='Processing dogs'):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(dog_folder, filename))
        if image is not None:
            image = cv2.resize(image, (100, 100))  # 调整图像大小
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序
            dog_images.append(image.flatten())  # 将图像转换为扁平的一维数组
            dog_labels.append(1)  # 1表示狗

# 读取猫图像文件并创建标签
cat_images = []
cat_labels = []
for filename in tqdm(os.listdir(cat_folder), desc='Processing cats'):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(cat_folder, filename))
        if image is not None:
            image = cv2.resize(image, (100, 100))  # 调整图像大小
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序
            cat_images.append(image.flatten())  # 将图像转换为扁平的一维数组
            cat_labels.append(0)  # 0表示猫

# 创建特征矩阵和标签向量
X = np.concatenate((dog_images, cat_images))
y = np.concatenate((dog_labels, cat_labels))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# 创建支持向量机分类器
classifier = SVC()

# 训练分类器
with tqdm(total=10, desc='Training classifier', unit='iteration') as pbar:
    for i in range(10):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        pbar.set_postfix({'Iteration': i+1, 'Accuracy': accuracy})
        pbar.update(1)


# 在测试集上进行预测
y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率: {:.2f}%".format(accuracy * 100))
