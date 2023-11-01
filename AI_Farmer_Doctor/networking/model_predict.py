import torch
from for_BERT import doubletext_to_encode
from fine_tuning_model import Fine_Net


def predict(data_1, data_2):
    output_encode = doubletext_to_encode(data_1, data_2, flag=102, max_len=10)  # Bert编码
    predict_output = fine_net(output_encode)  # 预测
    _, predicted_result = torch.max(predict_output, 1)  # 返回每一行最大值元素的索引，作为预测结果
    return str(predicted_result.item())  # 返回str类型的预测结果


if __name__ == '__main__':
    MODEL_PATH = "./model/Fine_BERT.pth"
    embedding_size = 768
    dropout = 0.2
    fine_net = Fine_Net(embedding_size, dropout)
    fine_net.load_state_dict(torch.load(MODEL_PATH))  # 加载模型权重
    fine_net.eval()  # 不进行梯度更新
    data_1 = "我家田地里面水稻苗枯萎"
    data_2 = "今天天气真好"
    print("预测结果:", predict(data_1, data_2))
