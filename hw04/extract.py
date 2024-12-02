import cv2
import numpy as np
import os
import sys

def scanimgfile(path):
    filelist = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            filelist.append(os.path.join(path, file))  # 使用完整路径
    return filelist

def processimg(imgname, threshold_value=60, kernel_size=(25, 25)):
    oriimg = cv2.imread(imgname)
    grayimg = cv2.cvtColor(oriimg, cv2.COLOR_BGR2GRAY)
    _, binaryimg = cv2.threshold(grayimg, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones(kernel_size, np.uint8)
    openimg = cv2.morphologyEx(binaryimg, cv2.MORPH_OPEN, kernel)

    rows, cols = openimg.shape 
    for row in range(rows):
        for col in range(cols):
            if openimg[row, col] == 0:
                oriimg[row, col] = (0, 0, 0)

    return oriimg

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[-1] == "":
        print("请提供图片文件夹路径")
        sys.exit(1)

    output_dir = "./afterprocess"
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    imglist = scanimgfile(sys.argv[-1])
    for img in imglist:
        res = processimg(img)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img)), res)  # 保存处理后的图片
