import cv2
import numpy as np

video_file = 'video/distance.mp4'
cap = cv2.VideoCapture(video_file)

# 初始化 ORB 特征检测器
orb = cv2.ORB_create()
prev_kp = None
prev_des = None
prev_frame = None
prev_pose = np.eye(4)  # 初始位姿
total_distance = 0  # 累加总距离

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测特征点和描述子
    kp, des = orb.detectAndCompute(gray, None)

    if prev_kp is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)

        # 提取关键点
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 计算位姿
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=2140, pp=[8.058856128102105e+02,1.286187934232550e+02], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

        # 更新
        translation = t.flatten()
        current_pose = np.dot(prev_pose, np.array([[R[0, 0], R[0, 1], R[0, 2], translation[0]],
                                                   [R[1, 0], R[1, 1], R[1, 2], translation[1]],
                                                   [R[2, 0], R[2, 1], R[2, 2], translation[2]],
                                                   [0, 0, 0, 1]]))

        # 计算两帧之间的相机移动距离
        distance = np.linalg.norm(current_pose[:3, 3])
        total_distance += distance

        # 可视化匹配
        frame_matches = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches, None)
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matches', 800, 600)
        cv2.imshow('Matches', frame_matches)

        # 更新位姿
        prev_pose = current_pose.copy()

    # 更新前一帧
    prev_frame = frame
    prev_kp, prev_des = kp, des

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


actual_distance = total_distance * 1.9/100

# 输出总移动距离
print("Total camera displacement distance:", total_distance)
print("Actual distance:", actual_distance,"cm")

cap.release()
cv2.destroyAllWindows()