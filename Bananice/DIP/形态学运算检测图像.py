import cv2

image = cv2.imread("jianzhu.jpg", cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate_img = cv2.dilate(image, kernel)
erode_img = cv2.erode(image, kernel)
absdiff_img = cv2.absdiff(dilate_img, erode_img);  # 将两幅图像相减获得边缘
# 上面得到的结果是灰度图，将其二值化以便观察结果
retval, threshold_img = cv2.threshold(absdiff_img, 40, 255, cv2.THRESH_BINARY);
# 反色，对二值图每个像素取反
result = cv2.bitwise_not(threshold_img)
cv2.imshow("origin_img", image)
cv2.imshow("dilate_img", dilate_img)
cv2.imshow("erode_img", erode_img)
cv2.imshow("absdiff_img", absdiff_img)
cv2.imshow("threshold_img", threshold_img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
