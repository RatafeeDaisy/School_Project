import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread('../Dataset/images/bi.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 二值化
ret1, thresh = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 去噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                           kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 确定前景区域
dst = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret2, sure_fg = cv2.threshold(dst, 0.005 * dst.max(), 255, 0)

# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknowen = cv2.subtract(sure_bg, sure_fg)

# 类别标记
ret3, markers = cv2.connectedComponents(sure_fg)

# 分水岭分割
img = cv2.watershed(image, markers)
plt.subplot(121)
plt.title('origin image')
plt.imshow(imagergb)
plt.axis('off')
plt.subplot(122)
plt.title('watershed image')
plt.imshow(img)
plt.axis('off')
plt.show()
