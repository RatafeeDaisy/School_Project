# 导入Flask框架工具包
from flask import Flask

# 创建实例app,内部需要一个必要参数__name__
app = Flask(__name__)  # 使用装饰器传给Flask，让其拿到url地址


@app.route('/')
def for_flask():
    # 访问浏览器后，输出信息:”This is my Flask ”
    return "This is my Flask!"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8084)
