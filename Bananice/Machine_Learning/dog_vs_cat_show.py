import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载训练好的模型
model_path = 'D:/School/School_Project/Bananice/Dataset/model.h5'
model = keras.models.load_model(model_path)

# 指定文件夹路径
image_folder_path = 'D:/School/School_Project/Bananice/Dataset/show'  # 指定包含猫狗混合图片的文件夹路径
class_names = ['狗', '猫']  # 类别名称

# 获取文件夹中的图片文件列表
image_files = os.listdir(image_folder_path)

# 从文件列表中随机选择6张图片
random.shuffle(image_files)
selected_images = image_files[:6]

# 创建子图并绘制图片及预测结果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('猫狗分类预测示例', fontsize=16)

for i, image_name in enumerate(selected_images):
    image_path = os.path.join(image_folder_path, image_name)
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    # 预测图片类别
    predictions = model.predict(image_array)
    predicted_class = class_names[int(predictions[0][0] > 0.5)]

    # 显示图片和预测结果
    ax = axes[i // 3, i % 3]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'预测结果：{predicted_class}', fontsize=12)

plt.tight_layout()
plt.show()
