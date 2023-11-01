import requests  # 模拟用户先后输入的两句话

post_information = {"data_1": "我家田地里面水稻苗枯萎", "data_2": "我很好"}  # POST请求方式，URL地址加用户的两句话
outputs = requests.post("http://127.0.0.1:8084/farmer/receive/", data=post_information)
print("first data:", post_information["data_1"], "\t" + "second data:", post_information["data_2"])
print("result:", outputs.text)
