from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from matplotlib.colors import ListedColormap

cancer = load_breast_cancer()  # 加载数据集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
print(cancer.target)
svc = SVC()  # 创建SVC支持向量机分类器模型实例
svc.fit(X_train, y_train)  # 训练数据集
print('训练集得分：', svc.score(X_train, y_train))  # 输出预测的训练集与测试集评分
print('预测集得分：', svc.score(X_test, y_test))

plt.plot(X_train.min(axis=0), 'o', label='min')  # 找出每一列的最小值
plt.plot(X_train.min(axis=1), '^', label='max')  # 找出每一行的最小值
plt.legend()
plt.xlabel('Feature index')
plt.xlabel('Feature magnitude')
plt.yscale('log')
plt.show()

min_on_training = X_train.min(axis=0)  # 计算训练集中每个特征的最小值
range_on_training = (X_train - min_on_training).max(axis=0)  # 计算训练集中每个特征的范围(最大值-最小值)
# 减去最小值，然后除以范围，这样数据集的每个特征都会在0~1之间
X_train_scaled = (X_train - min_on_training) / range_on_training
X_train_scaled  # 输出所有进行缩放后的数据
print('每个特征的最小值\n{}'.format(X_train_scaled.min(axis=0)))
print('每个特征的最大值\n{}'.format(X_train_scaled.max(axis=0)))

X_test_scaled = (X_test - min_on_training) / range_on_training
svc2 = SVC()  # 定义SVC支持向量机分类器
svc2.fit(X_train_scaled, y_train)  # 训练进行缩放后的数据集
print('缩放后训练集得分：', svc2.score(X_train_scaled, y_train))
print('缩放后测试集得分：', svc2.score(X_test_scaled, y_test))

svc3 = SVC(C=30)  # 定义SVC支持向量机分类器
svc3.fit(X_train_scaled, y_train)  # 训练进行缩放后的数据集
print('正则化后训练集得分：', svc3.score(X_train_scaled, y_train))
print('正则化后测试集得分：', svc3.score(X_test_scaled, y_test))


def f(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1))
    # linspace在指定范围内生成序列，reshape(-1,1)转换成1列：
    x_new = np.c_[x0.ravel(), x1.ravel()]  # ravel将多维数组降位一维
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    y_predict = model.predict(x_new)  # 获取预测值
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D'])  # 设置区域颜色
    plt.contourf(x0, x1, zz, cmap=custom_cmap)  # 绘制等高线


svc4 = SVC(C=30)  # 定义SVC支持向量机分类器
svc4.fit(X_train_scaled[:, 0:2], y_train)  # 训练进行缩放后的数据集
f(svc4, axis=[0, 0.8, 0, 1.3])  # 调用方法绘制边界
plt.scatter(X_test_scaled[y_test == 0, 0], X_test_scaled[y_test == 0, 1])  # 选取y所有为0的+X的第一列和第二列
plt.scatter(X_test_scaled[y_test == 1, 0], X_test_scaled[y_test == 1, 1])  # 选取y所有为1的+X的第一列
plt.show()
