import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    # 加载数据到DataFrame
    data = pd.read_csv(file_path)
    return data


def handle_outliers(data):
    # 处理离群值
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    upper = q3 + IQR * 1.5
    lower = q1 - IQR * 1.5
    upper_dict = dict(upper)
    lower_dict = dict(lower)

    for i, v in data.items():
        v_col = v[(v <= lower_dict[i]) | (v >= upper_dict[i])]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("列 {} 的离群值 = {} => {}%".format(i, len(v_col), round(perc, 3)))


def perform_log_transform(data):
    # 使用对数转换
    data["age"] = np.log(data.age)
    data["trtbps"] = np.log(data.trtbps)
    data["chol"] = np.log(data.chol)
    data["thalachh"] = np.log(data.thalachh)
    print("---已执行对数转换---")


def preprocess_data(data):
    # 预处理数据
    # 将特征和目标变量分离
    X = data.drop("output", axis=1)
    y = data["output"]

    # 特征标准化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    return X, y


def train_model(X, y):
    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 训练K近邻分类器模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def evaluate_model(model, X_test, y_test):
    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 输出分类报告
    print(metrics.classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    plot_confusion_matrix(model, X_test, y_test)


def plot_confusion_matrix(model, X_test, y_test):
    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算混淆矩阵
    cm = metrics.confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    # 加载数据
    file_path = "D:/School/School_Project/Bananice/Dataset/heart.csv"  # 替换为实际数据文件的路径
    data = load_data(file_path)

    # 处理离群值
    handle_outliers(data)

    # 执行对数转换
    perform_log_transform(data)

    # 预处理数据
    X, y = preprocess_data(data)

    # 训练和评估模型
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
