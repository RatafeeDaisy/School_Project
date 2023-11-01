import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")


def doubletext_to_encode(text_1, text_2, flag=102, max_len=10):
    bert_tokens = tokenizer.encode(text_1, text_2)  # 将两句文本根据词典编码为数字，称之为token embedding.分隔符对应的编码为102
    f = bert_tokens.index(flag)  # 找到分隔符的位置，方便下面进行长度规范
    # 规范第一句话的文本长度
    if len(bert_tokens[:f]) >= max_len:
        standard_bert_tokens1 = bert_tokens[:max_len]
    else:
        standard_bert_tokens1 = bert_tokens[:f] + (max_len - len(bert_tokens[:f])) * [0]
    # 规范第二句话的文本长度
    if len(bert_tokens[f:]) >= max_len:
        standard_bert_tokens2 = bert_tokens[f:f + max_len]
    else:
        standard_bert_tokens2 = bert_tokens[f:] + (max_len - len(bert_tokens[f:])) * [0]
    final_bert_tokens = standard_bert_tokens1 + standard_bert_tokens2  # 将统一长度后的两句文本的编码进行拼接
    ids = [0] * max_len + [1] * max_len  # segment embedding，前一句对应元素的编码为0，那么后一句编码为1
    intensor_ids = torch.tensor([ids])  # 转化为Tensor
    intensor_tokens = torch.tensor([final_bert_tokens])  # 转化为Tensor
    with torch.no_grad():
        pooled_output, _ = model(intensor_tokens, token_type_ids=intensor_ids, return_dict=False)  # 最终的编码结果
    return pooled_output


# 定义主函数，测试上述代码的有效性：
if __name__ == '__main__':
    data1 = "春江潮水连海平"
    data2 = "海上明月共潮生"
    encoded_layers = doubletext_to_encode(data1, data2)
    print(encoded_layers)
    print(encoded_layers.shape)
