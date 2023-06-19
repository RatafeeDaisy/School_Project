import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect(image):
    # 创建SIFT生成器，descriptor为一个对象的描述符
    descriptor = cv2.xfeatures2d.SIFT_create()

    # 检测特征点及其描述子（128维向量）
    kps, features = descriptor.detectAndCompute(image, None)
    return kps, features


# 定义查看特征点情况的函数——show_points，
def show_points(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)
    print(f"特征点数：{len(kps)}")
    img_left_points = cv2.drawKeypoints(image, kps, image)
    plt.figure(), plt.axis('off')
    plt.imshow(img_left_points)


# 查看图像中检测到的特征点，运行下面的第一步主程序部分，显示结果如图所示，输出的特征点数：4971

# 第一步主程序部分
if __name__ == '__main__':
    img_left = cv2.imread('images/IMGL.png', 1)
    img_right = cv2.imread('images/IMGR.png', 1)
    plt.subplot(121), plt.axis('off')
    plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.axis('off')
    plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    show_points(img_left)
    plt.show()


def match_keypoints(kps_left, kps_right, features_left, features_right, ratio, threshold):

    # 创建暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")

    # 使用KNN检测，匹配left、right图的特征点
    raw_matches = matcher.knnMatch(features_left, features_right, 2)
    print(len(raw_matches))
    matches = []  # 记录坐标
    good = []  # 记录特征点
    for m in raw_matches:  # 筛选匹配点
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:  # 筛选条件
            good.append([m[0]])
            matches.append((m[0].queryIdx, m[0].trainIdx))

    # 特征点对数大于4，就可以构建变换矩阵
    kps_left = np.float32([kp.pt for kp in kps_left])
    kps_right = np.float32([kp.pt for kp in kps_right])
    print(len(matches))
    if len(matches) > 4:

        # 获取匹配点坐标
        pts_left = np.float32([kps_left[i] for (i, _) in matches])
        pts_right = np.float32([kps_right[i] for (_, i) in matches])

        # 计算变换矩阵H
        H, status = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, threshold)
        return matches, H, good
    return None


if __name__ == '__main__':
    img_left = cv2.imread('images/IMGL.png', 1)
    img_right = cv2.imread('images/IMGR.png', 1)
    kps_left, features_left = detect(img_left)
    kps_right, features_right = detect(img_right)
    matches, H, good = match_keypoints(kps_left, kps_right, features_left, features_right, 0.5, 0.99)
    img = cv2.drawMatchesKnn(img_left, kps_left, img_right, kps_right, good[:30], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 20)), plt.axis('off')
    plt.imshow(img)
    plt.show()


def drawMatches(img_left, img_right, kps_left, kps_right, matches, H):

    # 获取图像宽度和高度
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    image = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
    image[0:h_left, 0:w_left] = img_right

    # 利用以获得的单应性矩阵进行变透视换
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

    # 将透视变换后的图像与另一张图像进行拼接
    image[0:h_left, 0:w_left] = img_left
    return image


if __name__ == '__main__':
    img_left = cv2.imread('images/IMGL.png', 1)
    img_right = cv2.imread('images/IMGR.png', 1)

    # 模块一：提取特征
    kps_left, features_left = detect(img_left)
    kps_right, features_right = detect(img_right)

    # 模块二：特征匹配
    matches, H, good = match_keypoints(kps_left, kps_right, features_left, features_right, 0.5, 0.99)

    # 模块三：透视变换-拼接
    vis = drawMatches(img_left, img_right, kps_left, kps_right, matches, H)

    # 显示拼接图形
    plt.figure(), plt.axis('off')
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.show()
