import cv2
import numpy as np
import glob
# 准备标定板的世界坐标（例如，棋盘格）
chessboard_size = (11, 8)  # 内部角点数
square_size = 0.025  # 每个方块的实际尺寸（米）

# 准备对象点和图像点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

obj_points = []  # 3D 点
img_points = []  # 2D 点

# 读取标定图像
images = glob.glob('chess/*.jpg')  # 假设图像格式为 jpg，修改为实际格式

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 输出内参
print("Camera matrix:\n", mtx)

