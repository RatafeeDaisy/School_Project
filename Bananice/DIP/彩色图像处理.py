import cv2

img = cv2.imread('d:/pics/lena.jpg')

# 全局阈值中的二值化阈值处理的彩色图像
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 采用权重相等方式的局部阈值处理的彩色图像
img_b, img_g, img_r = cv2.split(img)
ath2_b = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 5, 3)
ath2_g = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 5, 3)
ath2_r = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 5, 3)
mean_ada_img = cv2.merge([ath2_b, ath2_g, ath2_r])  # 合并通道
ath3_b = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 5, 3)
ath3_g = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 5, 3)
ath3_r = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 5, 3)
gauss_ada_img = cv2.merge([ath3_b, ath3_g, ath3_r])  # 合并通道

cv2.imshow('origin_img', img)
cv2.imshow('threshold_img', th1)
cv2.imshow('mean_ada_thr', mean_ada_img)
cv2.imshow('gauss_ada_thr', gauss_ada_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
