import cv2
import imutils
import numpy as np

img = cv2.imread('images/chepai1.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (600, 400))
cv2.imshow('Origin', img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 双边滤波
img_gray = cv2.bilateralFilter(img_gray, 13, 15, 15)
img_edged = cv2.Canny(img_gray, 30, 200)
cv2.imshow('edged', img_edged)

# 寻找轮廓，三个输入参数：输入图像，轮廓检索方式，轮廓近似方法
img_contours = cv2.findContours(img_edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

# 返回countors中的轮廓
img_contours = imutils.grab_contours(img_contours)
img_contours = sorted(img_contours, key=cv2.contourArea,
                      reverse=True)[:10]
# print('contours',img_contours)
screenCnt = None

for c in img_contours:
    peri = cv2.arcLength(c, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)  # 多边形拟合曲线
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print('No contour detected')
else:
    detected = 1
if detected == 1:
    cv2.drawContours(img, [screenCnt], -1,
                     (0, 0, 255), 3)
mask = np.zeros(img_gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
cv2.imshow('mask_image', new_image)  # 找到车牌位置，并掩模
new_image = cv2.bitwise_and(img, img, mask=mask)  # 与原图像“与”操作
# cv2.imshow('bitwisenew_image', new_image)
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))

# 切割
cropped = img_gray[topx:bottomx + 1, topy:bottomy + 1]
cropped = cv2.resize(cropped, (400, 200))
cv2.imshow('Cropped', cropped)
cv2.waitKey()
cv2.destroyAllWindows()
