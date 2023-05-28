import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# 导入数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据进行归一化，方便计算;参数-1就是不知道行数或者列数多少的情况下使用的参数
x_train = x_train.reshape((-1, 28 * 28)) / 255.0
x_test = x_test.reshape((-1, 28 * 28)) / 255.0

# 设置中间层的神经元个数
code_dim1 = 256
code_dim2 = 128
code_dim3 = 64
code_dim4 = 32

# 建立编码器的网络结构,模型堆叠
encoder = tf.keras.Sequential()
encoder.add(layers.Input(shape=(x_train.shape[1],), name='inputs'))  # 输入层
encoder.add(layers.Dense(code_dim1, activation='relu', name='encoder1'))
encoder.add(layers.Dense(code_dim2, activation='relu', name='encoder2'))
encoder.add(layers.Dense(code_dim3, activation='relu', name='encoder3'))
encoder.add(layers.Dense(code_dim4, activation='relu', name='encoder4'))

# 解码器层：
encoder.add(layers.Dense(code_dim4, activation='relu', name='decoder1'))
encoder.add(layers.Dense(code_dim3, activation='relu', name='decoder2'))
encoder.add(layers.Dense(code_dim2, activation='relu', name='decoder3'))
encoder.add(layers.Dense(code_dim1, activation='relu', name='decoder4'))
encoder.add(layers.Dense(x_train.shape[1], activation='softmax',
                         name='decoder_output'))
print(encoder.summary())

# 配置编码器的训练过程：
encoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='binary_crossentropy', metrics='accuracy')
history = encoder.fit(x_train, x_train, batch_size=64, epochs=50,
                      validation_split=0.1)
decoded = encoder.predict(x_test)

# 显示图片
n = 5
plt.figure(figsize=(10, 4))  # 整个图像的大小
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 绘制训练准确率和测试的准确率
metric = 'accuracy'
train_metric = history.history[metric]
val_metric = history.history['val_' + metric]
epochs = range(1, len(train_metric) + 1)
plt.figure(num='figure2')
plt.plot(epochs, train_metric, 'bo--')
plt.plot(epochs, val_metric, 'ro--')
plt.title('Training and validation' + metric)
plt.xlabel("Epochs")
plt.ylabel(metric)
plt.legend(["train_" + metric, 'val_' + metric])
plt.show()