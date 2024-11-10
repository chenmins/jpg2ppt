import cv2
import numpy as np
import os


def detect_screen(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 调整高斯模糊的核大小
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # 从 (5, 5) 调整到 (9, 9)

    # 调整 Canny 边缘检测器的阈值
    edges = cv2.Canny(blurred, 100, 200)  # 将阈值从 50, 150 调整到 100, 200

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到面积最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        img_cropped = crop_min_area_rect(img, rect)

        # 等比缩放图像为原来的50%
        img_resized = cv2.resize(img_cropped, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite(output_path, img_resized)


def crop_min_area_rect(img, rect):
    # 获取旋转矩阵
    center, size, angle = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotated = cv2.warpAffine(img, M, (width, height))

    img_cropped = cv2.getRectSubPix(img_rotated, size, center)
    return img_cropped


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            detect_screen(input_path, output_path)


# 调用函数处理目录中的所有图片
process_images('d:/jpg', 'd:/newjpg')
