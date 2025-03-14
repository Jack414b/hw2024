import cv2
import numpy as np

# 读取图片
img1 = cv2.imread('image3.jpg')
img2 = cv2.imread('image4.jpg')

# 转换为灰度图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Harris 角点检测
corners1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
corners2 = cv2.cornerHarris(gray2, 2, 3, 0.04)

# 调整阈值并提取角点
threshold1 = 0.01 * corners1.max()
threshold2 = 0.01 * corners2.max()

keypoints1 = np.argwhere(corners1 > threshold1)
keypoints2 = np.argwhere(corners2 > threshold2)

# 匹配角点特征
matches = []
max_matches = 100
for kp1 in keypoints1:
    if len(matches) >= max_matches:
        break  
    best_match = None
    best_distance = float('inf')
    for kp2 in keypoints2:
        if 0 <= kp1[0] < img1.shape[0] and 0 <= kp1[1] < img1.shape[1] and \
           0 <= kp2[0] < img2.shape[0] and 0 <= kp2[1] < img2.shape[1]:
            distance = np.sum((img1[kp1[0], kp1[1]] - img2[kp2[0], kp2[1]]) ** 2)  # SSD
            if distance < best_distance:
                best_distance = distance
                best_match = kp2
    if best_match is not None:
        matches.append((kp1, best_match))

# 绘制匹配结果
def draw_matches(img1, img2, matches):
    matched_image = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    matched_image[:img1.shape[0], :img1.shape[1]] = img1
    matched_image[:img2.shape[0], img1.shape[1]:] = img2

    for (kp1, kp2) in matches:
        pt1 = (kp1[1], kp1[0])  # keypoint1 的坐标
        pt2 = (kp2[1] + img1.shape[1], kp2[0])  # keypoint2 的坐标
        cv2.line(matched_image, pt1, pt2, (0, 255, 0), 1)

    return matched_image

matched_image = draw_matches(img1, img2, matches)

cv2.namedWindow('Matched Features', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched Features', 800, 600) 

# 显示结果
cv2.imshow('Matched Features', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('matched_features.jpg', matched_image)




