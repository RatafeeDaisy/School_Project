import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as linear_model
from sklearn.preprocessing import PolynomialFeatures

#加载训练数据，建立回归方程
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
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)

#可视化处理
plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),  color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

#模型评估
print("回归系数:",lin_reg_2.coef_) #回归系数
print("截距:",lin_reg_2.intercept_) #截距

