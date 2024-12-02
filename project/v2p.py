import cv2
import os

# 视频文件路径
video_path = 'chessboard.mp4'
# 输出图片的文件夹
output = 'extracted_frames'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output):
    os.makedirs(output)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 帧计数器
frame_count = 0

while True:
    # 读取帧
    ret, frame = cap.read()
    
    # 如果没有读取到帧，结束循环
    if not ret:
        break
    
    # 保存帧为图片
    frame_filename = os.path.join(output, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    # 增加帧计数
    frame_count += 1

# 释放视频对象
cap.release()

print(f'提取完成，共提取了 {frame_count} 帧图片。')
