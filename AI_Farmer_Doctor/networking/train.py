import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from for_BERT import doubletext_to_encode
from fine_tuning_model import Fine_Net

device = torch.device('cuda:0')


def load_data(path, batch_size, split=0.2):
    whole_data = pd.read_csv(path, header=None, sep="\t")  # 读取数据
    whole_data = shuffle(whole_data).reset_index(drop=True)  # 打乱数据
    print("正样本数量:" + str(dict(Counter(whole_data[0].values))[1]))  # 查看正样本的数量
    print("负样本数量:" + str(dict(Counter(whole_data[0].values))[0]))  # 查看负样本的数量
    # 划分数据集，训练集 ：验证集 = 4 ：1
    split_point = int(len(whole_data) * split)
    valid_data = whole_data[:split_point]
    train_data = whole_data[split_point:]

    # 定义数据迭代器
    def data_generator(data):
        # 单批次的数据生成函数
        for i in range(0, len(data), batch_size):
            encoded_talkdata_list = []  # 定义一个空列表，用于存放BERT编码后的数据
            encoded_label_list = []  # 用于存放BERT编码后的标签
            for j in data[i: i + batch_size].values.tolist():
                talk_data = doubletext_to_encode(j[1], j[2])  # BERT编码
                encoded_talkdata_list.append(talk_data)
                encoded_label_list.append([int(j[0])])
            talkdata = reduce(lambda x, y: torch.cat((x, y), dim=0),
                              encoded_talkdata_list)  # 利用reduce函数将列表中的数据转化成(batch_size,2*max_len,embedding_size)形式的张量
            labels = torch.tensor(reduce(lambda x, y: x + y, encoded_label_list))  # 将列表中的标签数据转化为tensor数据形式
            yield (talkdata, labels)

    return data_generator(train_data), len(train_data), data_generator(valid_data), len(
        valid_data)  # 处理后的训练集，训练集的数据量，处理后的验证集，验证集的数据量


fine_net = Fine_Net(768)  # 实例化分类网络模型
fine_net = fine_net.to(device)
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
loss_function = loss_function.to(device)
optimizer = optim.SGD(fine_net.parameters(), lr=0.001, momentum=0.9)  # 优化器，随机梯度下降


def train(train_data_all):
    train_whole_loss = 0.0  # 初始化损失值
    train_whole_acc = 0.0  # 初始化准确率
    # 参数更新
    for train_data, train_label in train_data_all:  # 遍历训练数据
        train_data = train_data.to(device)
        train_label = train_label.to(device)
        optimizer.zero_grad()  # 优化器梯度归零
        train_output = fine_net(train_data)  # 分类模型的输出
        train_loss = loss_function(train_output, train_label)  # 当前批次的损失
        train_whole_loss = train_whole_loss + train_loss.item()  # 总损失
        train_loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        train_whole_acc = train_whole_acc + (train_output.argmax(1) == train_label).sum().item()  # 总的准确率
    return train_whole_loss, train_whole_acc


def valid(valid_data_all):
    valid_whole_loss = 0.0  # 初始化验证损失值
    valid_whole_acc = 0.0  # 初始化验证准确率
    for valid_data, valid_label in valid_data_all:  # 遍历验证数据
        valid_data = valid_data.to(device)
        valid_label = valid_label.to(device)
        with torch.no_grad():  # 不进行梯度更新
            valid_output = fine_net(valid_data)  # 分类模型的输出
            valid_loss = loss_function(valid_output, valid_label)  # 计算当前批次的损失
            valid_whole_loss = valid_whole_loss + valid_loss.item()  # 总的验证损失
            valid_whole_acc = valid_whole_acc + (valid_output.argmax(1) == valid_label).sum().item()  # 总的验证准确率
    return valid_whole_loss, valid_whole_acc


if __name__ == '__main__':
    # 训练
    path = "./dataset/train_data.csv"  # 数据集路径
    epochs = 10  # 训练轮次
    batch_size = 128  # 每次喂给模型64条数据
    trainloss_list = []
    trainacc_list = []
    validacc_list = []
    validloss_list = []
    # 进行指定轮次的训练
    for i in range(epochs):
        # 打印轮次
        print("轮次:", i + 1)
        train_data_all, train_data_len, valid_data_all, valid_data_len = load_data(path, batch_size)  # 数据集划分及处理
        train_whole_loss, train_whole_acc = train(train_data_all)  # 训练
        valid_whole_loss, valid_whole_acc = valid(valid_data_all)  # 验证
        train_average_loss = train_whole_loss * batch_size / train_data_len  # 训练的平均损失
        train_average_acc = train_whole_acc / train_data_len  # 训练的平均准确率
        valid_average_loss = valid_whole_loss * batch_size / valid_data_len  # 验证的平均损失
        valid_average_acc = valid_whole_acc / valid_data_len  # 验证的平均准确率
        trainloss_list.append(train_average_loss)
        validloss_list.append(valid_average_loss)
        trainacc_list.append(train_average_acc)
        validacc_list.append(valid_average_acc)
        print("成功开始训练模型！")
        print("训练时损失:", train_average_loss, "|", "训练时准确率:", train_average_acc)
        print("验证时损失:", valid_average_loss, "|", "验证时准确率:", valid_average_acc)
    # 绘制训练和验证Loss曲线图
    plt.figure(0)
    plt.plot(trainloss_list, label="Train Loss")
    plt.plot(validloss_list, color="red", label="Valid Loss")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # 横坐标的间隔
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(1, epochs)
    plt.legend(loc='upper left')
    plt.savefig("./process_record/model_loss.png")
    # 绘制训练和验证的accuracy曲线图
    plt.figure(1)
    plt.plot(trainacc_list, label="Train Acc")
    plt.plot(validacc_list, color="red", label="Valid Acc")
    # 横坐标的间隔
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(1, epochs)
    plt.legend(loc='upper left')
    plt.savefig("./process_record/model_acc.png")
    # 模型保存
    MODEL_PATH = './model/Fine_BERT.pth'
    torch.save(fine_net.state_dict(), MODEL_PATH)
