import fileinput
import os
from neo4j import GraphDatabase  # 配置neo4j的信息

NEO4J_CONFIG = {"uri": "bolt://localhost:7687",
                "auth": ("neo4j", "533553"),
                "encrypted": False
                }

driver = GraphDatabase.driver(**NEO4J_CONFIG)


def neo4j_load_data(path):
    file_list = os.listdir(path)  # 返回path路径下面的所有文件
    disease_list = list(map(lambda x: x.split(".")[0], file_list))  # 去除文件后缀，得到所有水稻疾病实体词的列表
    rice_symptom = []  # 定义空列表，用于存储所有水稻疾病的症状
    for disease_csv in file_list:  # 遍历所有水稻疾病的文件
        # 单个水稻疾病文件的全部症状
        symptom = list(map(lambda x: x.strip(),
                           fileinput.FileInput(os.path.join(path, disease_csv),
                                               openhook=fileinput.hook_encoded('utf-8', '')
                                               )
                           )
                       )
        # 当前水稻疾病的所有症状
        rice_symptom.append(symptom)  # 将水稻症状添加到列表中
    return dict(zip(disease_list, rice_symptom))  # 返回字典型数据，字典套列表的形式  {'水稻疾病':['症状1','症状2','症状3',....],....}


def neo4j_input(path):
    ricedisease_symptom = neo4j_load_data(path)  # 获取水稻疾病及症状实体词
    # 将水稻疾病和症状实体词、疾病和症状之间的关系写入数据库
    with driver.session() as session:
        for disease, symptom in ricedisease_symptom.items():
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" % disease
            session.run(cypher)

            for i in symptom:
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" % i
                session.run(cypher)
                cypher = "MATCH (a:Disease{name:%r}) MATCH (b:Symptom{name:%r}) WITH a,b MERGE(a)-[r:dis_to_sym]-(b)" % (
                disease, i)
                session.run(cypher)


if __name__ == '__main__':
    diseasepath = "./predict_result/"  # 水稻疾病文件存储的目录
    neo4j_input(diseasepath)
