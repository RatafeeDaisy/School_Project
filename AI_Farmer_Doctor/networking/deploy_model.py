from waitress import serve
from flask import Flask, request
import torch
from for_BERT import doubletext_to_encode
from fine_tuning_model import Fine_Net

app = Flask(__name__)

MODEL_PATH = "./model/Fine_BERT.pth"  # 模型路径
embedding_size = 768  # 词嵌入维度
dropout = 0.2  # 随机失活率

fine_net = Fine_Net(embedding_size, dropout)  # 实例化前馈神经网络
fine_net.load_state_dict(torch.load(MODEL_PATH))  # 加载训练好的模型权重
fine_net.eval()  # 设置eval()模式，不进行梯度更新


# 定义路由
@app.route('/farmer/receive/', methods=["POST"])
# 部署模型请求方式为POST形式，请求参数为/farmer/receive/
def receive():
    data_1 = request.form['data_1']  # 用户输入的第一句话
    data_2 = request.form['data_2']  # 用户输入的第二句话
    output_encode = doubletext_to_encode(data_1, data_2, flag=102, max_len=10)  # Bert-base-Chinese模型编码
    predict_output = fine_net(output_encode)  # 预测
    _, predicted_result = torch.max(predict_output, 1)  # 返回每一行最大值元素的索引，作为预测结果（0或1）

    return str(predicted_result.item())  # 返回str类型的预测结果


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8084)
