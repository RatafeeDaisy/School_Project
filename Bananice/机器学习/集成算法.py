import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std: object = sc.transform(X_test)


data = load_iris()  # 得到数据特征
iris_target = data.target  # 得到数据对应的类标签

'''2、特征预处理'''
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)  # 利用pandas转化为DataFrame格式

'''3、可视化数据特征分布'''
iris_all = iris_features.copy()  # 进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target  # 合并特征和类标签
# 特征与标签组合的散点可视化
sns.pairplot(data=iris_all, diag_kind='kde', hue='target')  # hue:使用指定变量为分类变量画图，这里使用target
plt.show()  # 从该图可以看出，仅凭petal length或petal width就可以把0类花区分出来

'''4、划分训练集、测试集'''
x_train, x_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.3, random_state=0)
# 测试集大小为30%

'''5、模型训练'''
# 定义逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')
# 在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)

'''6、模型测试'''
# 在训练集和测试集上分别利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

'''7、模型评估'''
# 查看混淆矩阵
M = metrics.confusion_matrix(test_predict, y_test)
print('混淆矩阵:\n', M)
# 利用热力图进行可视化
plt.figure()
sns.heatmap(M, annot=True, cmap='Blues')
plt.xlabel('predicted labels')
plt.ylabel('True labels')
plt.show()

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim = (xx2.min(), xx2.max())
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='black', alpha=0.8, linewidths=1, marker='o', s=10, label='test set')


tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined=np.vstack((X_train, X_test))
y_combined=np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='tree.dot',feature_names=['petal length', 'petal width']) # 导出为dot文件