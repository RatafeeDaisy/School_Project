import json
import random
import requests

my_id = "3e4GRa2XUmu4vT9PvaAgTWG8"  # ak
my_secret = "kwELFyAnO1X1dUy33roQVBNpngiZaISs"  # sk


def unit_chat(chat_input, user_id="68799"):
    # chat_input-->用户发送的聊天内容
    # user_id-->发起聊天用户ID，可自定义
    chat_reply = "不好意思，俺们正在学习中，随后回复你。"
    # 根据 my_id与 my_secret 获取access_token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (
        my_id, my_secret)
    res = requests.get(url)
    access_token = eval(res.text)["access_token"]
    # 请求的url地址
    ucu = "https://aip.baidubce.com/rpc/2.0/unit/service/chat?access_token=" + access_token
    post_data = {
        "log_id": str(random.random()),
        "request": {
            "query": chat_input,
            "user_id": user_id
        },
        "session_id": "",
        # 机器人的id
        "service_id": "S100661",
        # 技能id
        "bot_id": "1426127",
        "version": "3.0"
    }
    outputs = requests.post(url=ucu, json=post_data)
    # json解析url返回的结果
    uco = json.loads(outputs.content)
    if uco["error_code"] != 0:
        return chat_reply
    # ucor   承载结果
    # ucrl  unit返回的response列表
    ucor = uco["result"]
    ucrl = ucor["response_list"]
    ucro = random.choice(
        [ucr for ucr in ucrl if
         ucr["schema"]["intent_confidence"] > 0.0])
    # ucral    unit的response的action列表
    # ucrs    最终的unit结果
    ucral = ucro["action_list"]
    ucral_1 = random.choice(ucral)
    ucrs = ucral_1["say"]
    return ucrs


if __name__ == '__main__':
    while True:
        your_input = input("请输入:")
        unit_reply = unit_chat(your_input)
        print("Unit回答 >>>", unit_reply)
