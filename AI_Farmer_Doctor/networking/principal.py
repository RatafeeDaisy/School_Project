import requests
import redis
import json
from flask import Flask
from flask_cors import CORS
from flask import request
from for_Unit import unit_chat
from neo4j import GraphDatabase
from waitress import serve

app = Flask(__name__)
CORS(app, resources=r'/*')

# 配置Neo4j连接对象
NEO4J_CONFIG = {"uri": "bolt://localhost:7687", "auth": ("neo4j", "533553"), "encrypted": False
                }
neo4j_driver = GraphDatabase.driver(**NEO4J_CONFIG)

# 配置redis的挂起端口
REDIS_CONFIG = {
    "host": "127.0.0.1",
    "port": 6379
}
link_pool = redis.ConnectionPool(**REDIS_CONFIG)

subjectmodel_url = "http://127.0.0.1:8084/farmer/receive/"


def interrogate_inneo4j(input_content):
    with neo4j_driver.session() as session:
        find_inneo4j = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 6" % input_content
        output = list(map(lambda i: i[0], session.run(find_inneo4j)))
        return output


class Main_Logic(object):
    def __init__(self, id_for_user, input_content, rules_answer, redis_con):
        self.id_for_user = id_for_user
        self.input_content = input_content
        self.rules_answer = rules_answer  # 预先设置好的话，回复给用户

        self.redis_con = redis_con  # # 用于连接redis

    def foremost_deal_talk(self):
        # 用户发送的第一次会话，抓住水稻症状实体直接去查询对应的水稻病就可以了
        list_1 = interrogate_inneo4j(self.input_content)
        if not list_1:
            print("没有查询到该实体，访问百度Unit", self.input_content)
            return unit_chat(self.input_content)
        print("图数据库的查询结果为：", list_1)
        # redis存储用户的第一句话，进行会话管理
        self.redis_con.hset(str(self.id_for_user), "foremost_talk", str(list_1))
        self.redis_con.expire(str(self.id_for_user), 30000)
        # 把列表结果转化为字符串类型数据，组合成对话返回结果
        print("生成对话")
        str_talk = ",".join(list_1)
        return self.rules_answer["2"] % str_talk

    def second_deal_talk(self, foremost_sen):
        print("下面判断两句相关性：")
        try:
            # post请求两句话相关性判断模型，判断redis库中存储的用户的前一句话和本次输入的相关性
            result = requests.post(subjectmodel_url, data={"data_1": foremost_sen, "data_2": self.input_content},
                                   timeout=3)
            print("两句相关性为（1相关，0不相关）:", result.text)
            # 不相关则直接交给百度Unit兜底
            if result.text != "1":
                return unit_chat(self.input_content)
        except Exception as e:
            print("两句相关性请求失败:", e)
            return unit_chat(self.input_content)
        list_1 = interrogate_inneo4j(self.input_content)  # 根据症状实体查找症状实体
        print("图数据库的查询结果为：", list_1)
        if not list_1:
            print("没有查询到该实体，访问百度Unit", self.input_content)
            return unit_chat(self.input_content)
        # 第一次会话查知识图谱时查到的水稻疾病实体
        dis_foremost = self.redis_con.hget(str(self.id_for_user), "foremost_talk")
        print(dis_foremost)
        if dis_foremost:
            # 第二次对话查到的疾病dis_second
            dis_second = list(set(list_1) | set(eval(dis_foremost)))
            # 去掉第一次包含的疾病
            output = list(set(list_1) - set(eval(dis_foremost)))
        else:
            output = dis_second = list(set(list_1))
        # 会话管理
        self.redis_con.hset(str(self.id_for_user), "foremost_talk", str(dis_second))
        self.redis_con.expire(str(self.id_for_user), 30000)
        print("生成对话")
        # 如果新水稻疾病实体存在则按模板二返回结果，否则返回模板四"稻稻觉得还是稻稻之前为您判断的水稻病，并不是新的水稻病"
        if not output:
            return self.rules_answer["4"]
        else:
            output = ",".join(output)
            return self.rules_answer["2"] % output


@app.route('/farmer/receive/', methods=["POST"])  # 装饰器
def ser_principal():
    id_for_user = request.form['id_for_user']
    talk_user = request.form['talk_user']
    redis_con = redis.StrictRedis(connection_pool=link_pool)  # 连接redis
    foremost_talk_sym = redis_con.hget(str(id_for_user), "foremost_talk_sym")
    redis_con.hset(str(id_for_user), "foremost_talk_sym", talk_user)
    print("会话正常存储了对话")
    # 加载对话模板文件
    rules_answer = json.load(open("talk_template.json", "r" ,encoding="utf-8"))
    print('下面实例化Main_Logic类.')
    main_logic = Main_Logic(id_for_user, talk_user, rules_answer, redis_con)
    if foremost_talk_sym:
        return main_logic.second_deal_talk(foremost_talk_sym)
    else:
        return main_logic.foremost_deal_talk()


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
