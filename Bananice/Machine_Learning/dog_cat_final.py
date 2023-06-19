import os
import tensorflow as tf
from tensorflow import keras
import random
from shutil import copyfile
from keras.applications.vgg16 import VGG16

# 设置数据集路径
DIRPATH = 'D:/School/School_Project/Bananice/Dataset/PetImages'

# 打印目录名
for dirname in os.listdir(DIRPATH):
    print(dirname)

# 创建数据目录结构
os.mkdir('./data')
os.mkdir('./data/train')
os.mkdir('./data/valid')

# 遍历数据集中的文件夹
for folder in os.listdir(DIRPATH):
    files = os.listdir(os.path.join(DIRPATH, folder))
    images = []

    # 遍历文件夹中的文件
    for f in files:
        try:
            # 读取图像文件
            img = tf.io.read_file(os.path.join(DIRPATH, folder, f))
            img = tf.image.decode_image(img)

            # 如果图像是三维的，将其添加到图像列表中
            if img.ndim == 3:
                images.append(f)
        except:
            pass

    # 打乱图像列表
    random.shuffle(images)
    count = len(images)

    # 计算训练集和验证集的分割点
    split = int(0.8 * count)

    # 创建训练集和验证集的子目录
    os.mkdir(os.path.join('./data/train', folder))
    os.mkdir(os.path.join('./data/valid', folder))

    # 将图像复制到训练集目录
    for c in range(split):
        source_file = os.path.join(DIRPATH, folder, images[c])
        destination = os.path.join('./data/train', folder, images[c])
        copyfile(source_file, destination)

    # 将图像复制到验证集目录
    for c in range(split, count):
        source_file = os.path.join(DIRPATH, folder, images[c])
        destination = os.path.join('./data/valid', folder, images[c])
        copyfile(source_file, destination)

# 设置训练集和验证集的路径
train_path = './data/train'
valid_path = './data/valid'

# 创建训练集和验证集的数据生成器
train_batches = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=10
)

valid_batches = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=30
)

# 加载预训练的 VGG16 模型，并设置顶层结构
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结 VGG16 模型的所有层
for layer in base_model.layers:
    layer.trainable = False

# 获取 VGG16 模型的最后一层输出
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

# 添加自定义全连接层
x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(64, activation='relu', name='FC_1')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(2, activation='sigmoid', name='sigmoid')(x)

# 构建完整模型
VGG16_model = keras.Model(inputs=base_model.input, outputs=x)

# 编译模型
VGG16_model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=["accuracy"])

# 训练模型
VGG16_model.fit(train_batches, validation_data=valid_batches, epochs=10, steps_per_epoch=10)

# 保存完整模型
VGG16_model.save('D:/School/School_Project/Bananice/Dataset/model.h5')
