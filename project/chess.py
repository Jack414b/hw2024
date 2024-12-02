import cv2
import numpy as np

# 设置棋盘格的尺寸
chessboard_size = (4, 5)  # 内部角点数
square_size = 0.025  # 每个方格的实际大小（米）

# 准备对象点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储对象点和图像点
objpoints = []  # 3D 点
imgpoints = []  # 2D 点

# 读取视频文件
video_file = 'chessboard.mp4'  # 替换为您的视频文件路径
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔一定帧数处理
    if frame_count % 30 == 0:  # 每30帧提取一次
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 可视化角点
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', frame)
            cv2.waitKey(100)  # 等待100毫秒以查看结果
        else:
            print(f"Warning: Corners not found in frame {frame_count}.")  # 打印警告

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# 标定相机
if len(objpoints) > 0 and len(imgpoints) > 0:  # 确保有足够的点
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 输出相机内参
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
else:
    print("Error: Not enough points for calibration.")
