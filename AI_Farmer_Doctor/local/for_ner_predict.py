from pprint import pprint
from paddlenlp import Taskflow
import os

schema = ['水稻病症状']
anl_model = Taskflow("information_extraction", schema=schema, task_path='./uie_model/model_best')  # 训练好的模型
files = os.listdir("./dataset/unstructured_data")  # 要进行命名实体识别的数据目录

for file in files:  # 遍历所有的待识别数据文件
    with open(os.path.join("./dataset/unstructured_data", file), "r", encoding="UTF-8") as f, \
            open(os.path.join("./predict_result", file), "w", encoding="UTF-8") as g:
        data_1 = f.read()  # 读取每一个文件的非结构化数据
        data_1 = data_1.replace("\t", "", -1).replace("\n", "", -1)  # 去掉空行和空格
        # 利用模型进行命名实体识别
        result = anl_model(data_1)  # 进行命名实体识别，抽取水稻病症状实体词
        # 保存抽取的实体词
        for i in range(len(result[0]['水稻病症状'])):
            final_result = result[0]['水稻病症状'][i]['text']
            g.write(str(final_result))
            g.write("\n")
            print("end!!!!")
