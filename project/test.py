import cv2
import numpy as np

# 读取预录制的视频文件
video_file = 'test1.mp4'  
cap = cv2.VideoCapture(video_file)

# 初始化 ORB 特征检测器
orb = cv2.ORB_create()
prev_kp = None
prev_des = None
prev_frame = None
prev_pose = np.eye(4)  # 初始位姿

# 输出初始位姿
print("Initial Pose:\n", prev_pose)

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
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)

        # 提取匹配的关键点
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 计算相机位姿
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

        # 更新位姿
        translation = t.flatten()
        prev_pose = np.dot(prev_pose, np.array([[R[0, 0], R[0, 1], R[0, 2], translation[0]],
                                                 [R[1, 0], R[1, 1], R[1, 2], translation[1]],
                                                 [R[2, 0], R[2, 1], R[2, 2], translation[2]],
                                                 [0, 0, 0, 1]]))

        # 可视化匹配
        frame_matches = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches, None)
        cv2.imshow('Matches', frame_matches)

    # 更新前一帧
    prev_frame = frame
    prev_kp, prev_des = kp, des

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 输出结束时的位姿
print("Final Pose:\n", prev_pose)

# 计算相机位移距离
initial_translation = prev_pose[:3, 3]  # 初始位姿的平移部分
final_translation = prev_pose[:3, 3]  # 最终位姿的平移部分
distance = np.linalg.norm(final_translation - initial_translation)  # 计算欧几里得距离

print("Camera displacement distance:", distance)

cap.release()
cv2.destroyAllWindows()
