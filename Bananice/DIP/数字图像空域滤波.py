import cv2
import numpy as np
img = cv2.imread('lena.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
# 使用自定义的卷积函数
kernel3=np.array([[-1,-1,0], [-1,0,1],[0,1,1]])
kernel5=np.array([[-1,-1,-1,-1,0],[-1,-1,-1,0,1],[-1,-1,0,1,1], [-1,0,1,1,1], [0,1,1,1,1]])
image3=cv2.filter2D(img_gray,-1,kernel3)
image5=cv2.filter2D(img_gray,-1,kernel5)
cv2.imshow("Origin image",img_gray)  #原始图像
cv2.imshow("k3 image",image3)       #卷积核k3图像
cv2.imshow("k5 image",image5)       #卷积核k5图像
cv2.waitKey()
cv2.destroyAllWindows()
