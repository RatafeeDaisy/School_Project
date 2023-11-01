import torch
import torch.nn as nn
import torch.nn.functional as F


class Fine_Net(nn.Module):
    def __init__(self, embedding_size=768, dropout=0.2):
        super(Fine_Net, self).__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(20 * embedding_size, 8)  # 第一层全连接层
        self.fc2 = nn.Linear(8, 2)  # 第二层全连接层，二分类，所以输出神经元个数为2

    def forward(self, x):
        x = x.view(-1, 20 * self.embedding_size)  # 因为最大句子长度为10，字符数应该是两倍是20
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


if __name__ == '__main__':
    x = torch.randn(1, 20, 768)  # 模拟使用Bert-base-Chinese预训练模型编码两句文本的结果
    net = Fine_Net(768, 0.2)  # 实例化分类网络模型
    output = net(x)
    print(output)
