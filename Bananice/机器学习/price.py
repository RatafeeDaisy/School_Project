import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as linear
datasets_X = []
datasets_Y = []
fr = open('D:\\School\\Bananice\\数据集\\price.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split('\t')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
linear = linear.LinearRegression()
linear.fit(datasets_X,datasets_Y)
plt.scatter(datasets_X,datasets_Y,color='red')
plt.plot(X,linear.predict(X),color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
print('系数:',linear.coef_)
print('截距:',linear.intercept_)
