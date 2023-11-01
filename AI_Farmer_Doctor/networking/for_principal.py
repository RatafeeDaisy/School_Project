import requests  # 导入requests包，模拟post请求

# 模拟用户的id为13424，输入信息为"菌丝块"
output = requests.post("http://127.0.0.1:8084/farmer/receive/", data={"id_for_user": "13431", "talk_user": "菌丝块"})
print(output.text)
