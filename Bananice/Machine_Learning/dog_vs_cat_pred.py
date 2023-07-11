import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow import keras

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载保存的完整模型
saved_model = keras.models.load_model('D:/School/School_Project/Bananice/Dataset/model.h5')

# 加载验证集数据
valid_data_path = 'D:/School/School_Project/Bananice/Machine_Learning/data/valid'  # 验证集数据路径
valid_batches = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    valid_data_path,
    target_size=(224, 224),
    batch_size=30,
    shuffle=False
)

# 获取验证集数据的真实标签
true_labels = valid_batches.classes

# 进行模型预测
predictions = saved_model.predict(valid_batches)
predicted_labels = np.argmax(predictions, axis=1)

# 计算评估分数
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# 打印评估分数
print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
print("F1 Score:", f1)

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.xticks([0, 1], ['猫', '狗'])
plt.yticks([0, 1], ['猫', '狗'])
plt.show()

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(true_labels, predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC曲线 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()
