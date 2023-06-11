import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


diabetes = load_wine()
X, Y = diabetes.data, diabetes.target


# 1.使用MLPClassifier进行分类
scaler = StandardScaler()  # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)  # 转换数据集
clf_class = MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs', alpha=1e-5, random_state=1)
clf_class.fit(X, Y)
y_pred = clf_class.predict(X)
print('分类预测结果:', y_pred)
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_pred, label='Samples')
plt.show()


# 2.使用MLPRegressor进行回归
diabetes = load_wine()
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs', )
model = clf.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('回归预测结果：', y_pred)
print(clf.n_layers_)  # 总的层数
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
