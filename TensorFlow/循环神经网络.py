import tensorflow as tf
import numpy as np
import tensorflow as keras
from keras import losses, Sequential, optimizers, layers, datasets

batchsz = 128  # 批量大小
total_words = 10000  # 词汇表大小N_vocab
max_review_len = 80  # 句子最大长度 s，大于的句子部分将截断，小于的将填充
embedding_len = 100  # 词向量特征长度
# 加载IMDB数据集，此处的数据采用数据编码，一个数字代表一个单词
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
# 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = db_test.batch(batchsz, drop_remainder=True)


# 统计数据集属性
# print('x_train shape: ',x_train.shape,tf.reduce_max(y_train),tf.reduce_min(y_train))
# print('x_test shape: ',x_test.shape


class MyRNN(tf.keras.Model):
    # cell方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b,64],构建cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # 词向量编码[b,80]=>[b,80,100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # 构建2个cell，使用dropout技术防止过拟合
        self.run_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.run_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # self.outlayer=layers.Dense(1)
        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b,80,100]->[b,64]->[b,1]
        self.outlayer = Sequential([
            layers.Dense(units),
            layers.Dropout(rate=0.5),
            layers.ReLU(),
            layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs
        # 获取词向量[b,80]=>[b,80,100]
        x = self.embedding(x)
        # 通过2个RNN CELL，[b,80,100]=>[b,64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word:[b,100]
            out0, state0 = self.run_cell0(word, state0, training)
            out1, state1 = self.run_cell1(out0, state1, training)
        # 末层最后一个输出作为分类网络的输入：[6,64]=>[b,1]
        x = self.outlayer(out1, training)
        # 通过激活函数p(y is pos[x])
        prob = tf.sigmoid(x)
        return prob


def main():
    units = 64  # RNN状态向量长度n
    epochs = 20  # 训练epochs
    learning_rate = 0.001

    model = MyRNN(units)  # 创建模型
    # 装配
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    # 训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
