from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn import tree

def decision_iris():
    # 1）获取数据集
    iris = load_iris()

    # 2)划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)

    # 3)决策树预估器
    # criterion: 默认是’gini’系数，也可以选择信息增益的熵’entropy’
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4)模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 5)决策树可视化
    dot_data = tree.export_graphviz(estimator, out_file=None, feature_names=iris.feature_names)
    # 生成可视化pdf
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')
    return None

decision_iris()