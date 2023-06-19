import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    center_position = (image_size[0] - 1) / 2
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  #
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图像进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # 二维数组的傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result


if __name__ == '__main__':
    image = cv2.imread('D:/School/School_Project/Bananice/DIP/images/lena.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行运动模糊处理
    img_h, img_w = image.shape[0:2]
    PSF = motion_process((img_h, img_w), 60)
    blurred = np.abs(make_blurred(image, PSF, 1e-3))
    plt.subplot(221), plt.axis('off')
    plt.title("Motion blurred")
    plt.imshow(blurred)
    plt.imsave('lenaMb.jpg', blurred)

    resultwd = wiener(blurred, PSF, 1e-3)  # 维纳滤波
    plt.subplot(222), plt.axis('off')
    plt.title("wiener deblurred(k=0.01)")
    plt.imshow(resultwd)
    plt.imsave('lenaWd.jpg', resultwd)
    blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    plt.subplot(223), plt.axis('off')
    plt.title("motion & noisy blurred")
    plt.imshow(blurred_noisy)  # 显示添加噪声且运动模糊的图像

    # 对添加噪声的图像进行维纳滤波
    resultwdn = wiener(blurred_noisy, PSF, 0.1 + 1e-3)
    plt.subplot(224), plt.axis('off')
    plt.title("wiener deblurred(k=0.01)")
    plt.imshow(resultwdn)
    plt.show()
