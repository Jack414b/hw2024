import cv2
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

video_file = 'video/distance.mp4'
output_file = 'matches_output.mp4'
cap = cv2.VideoCapture(video_file)

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()
prev_kp = None
prev_des = None
prev_frame = None
prev_pose = np.eye(4)  # 初始位姿
total_distance = 0  # 累加总距离
initial_pose = prev_pose.copy() 

# 打印位姿函数
def print_pose_matrix(pose_matrix):
    for i in range(4):
        row = []
        for j in range(4):
            # 判断是旋转还是平移
            if i < 3 and j < 3:
                row.append(f"{Fore.YELLOW}{pose_matrix[i, j]:+.3e}{Style.RESET_ALL}")
            elif j == 3 and i < 3:
                row.append(f"{Fore.BLUE}{pose_matrix[i, j]:+.3e}{Style.RESET_ALL}")
            else:
                row.append(f"{pose_matrix[i, j]:+.3e}")
        print(" | ".join(row))


# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

# 输出位姿
print("Initial Pose:")
print_pose_matrix(initial_pose)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测特征点和描述子
    kp, des = sift.detectAndCompute(gray, None)

    if prev_kp is not None:
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_des, des)

        # 提取匹配的关键点
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 计算相机位姿
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, pp=[8.058856128102105e+02,1.286187934232550e+02], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is not None and mask is not None:
            points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
        else:
            print("无法计算基础矩阵。")

        # 更新位姿
        translation = t.flatten()
        current_pose = np.dot(prev_pose, np.array([[R[0, 0], R[0, 1], R[0, 2], translation[0]],
                                                 [R[1, 0], R[1, 1], R[1, 2], translation[1]],
                                                 [R[2, 0], R[2, 1], R[2, 2], translation[2]],
                                                 [0, 0, 0, 1]]))
        initial_pose += current_pose 

        # 计算两帧之间的相机移动距离
        distance = np.linalg.norm(current_pose[:3, 3])
        total_distance += distance
      
        # 可视化匹配
        frame_matches = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches, None)
        # 创建窗口并设置为小尺寸
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matches', 800, 600)
        cv2.imshow('Matches', frame_matches)

        # 将可视化的匹配结果写入视频文件
        out.write(frame_matches)

    # 更新前一帧
    prev_frame = frame
    prev_kp, prev_des = kp, des

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Final Pose:")
print_pose_matrix(initial_pose)

actual_distance = total_distance * 1.9
# 输出总移动距离
print("Total camera displacement distance:", total_distance)
print("Actual distance:", actual_distance, "cm")

# 释放资源
cap.release()
out.release()  # 释放 VideoWriter
cv2.destroyAllWindows()
