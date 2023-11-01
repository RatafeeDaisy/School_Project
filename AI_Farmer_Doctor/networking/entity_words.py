import os

file_root = '../local/predict_result/'  # 用uie模型抽取的“水稻病症状”实体保存文件的路径
filenames = os.listdir(file_root)  # 返回predict_result/目录下的所有文件名称
words = []
for file_name in filenames:  # 遍历所有的文件
    file_path = os.path.join(file_root, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        word = f.read().split()
        words = words + word  # 将读取到的“水稻病症状”实体词保存到一个列表中
    # 将所有的水稻病症状”实体词写入“all.txt”文件
    with open("dataset/all.txt", 'w', encoding='utf-8') as f:
        for i in words:
            f.write(i + '\n')
    f.close()
