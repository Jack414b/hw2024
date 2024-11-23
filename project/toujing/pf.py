from PIL import Image
import numpy as np
import cv2

# 加载图片
image_path = 'microlens_array_circles_2_to_1.png'  # 替换为你的图片路径
image = Image.open(image_path).convert("RGBA")  # 确保图像是RGBA格式

# 获取图片的宽度和高度
width, height = image.size

# 定义透视变换矩阵
src_points = np.array([
    [0, 0],              # 左上角
    [width, 0],         # 右上角
    [width, height],    # 右下角
    [0, height]         # 左下角
], dtype='float32')

# 目标点，创建透视效果
dst_points = np.array([
    [200, 75],          # 左上角（上方较近）
    [width - 190, 75],  # 右上角（上方较近）
    [width - 20, height],  # 右下角（下方较远）
    [20, height]        # 左下角（下方较远）
], dtype='float32')

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 创建一个透明背景的图像
transparent_background = np.zeros((height, width, 4), dtype=np.uint8)  # 4通道图像，包含 alpha 通道

# 应用透视变换
transformed_image = cv2.warpPerspective(np.array(image), matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

# 将变换后的图像合并到透明背景上
transparent_background[:, :, :3] = transformed_image[:, :, :3]  # RGB通道
transparent_background[:, :, 3] = (transformed_image[:, :, 3] > 0).astype(np.uint8) * 255  # 设置 alpha 通道

# 将结果转换回PIL格式
result_image = Image.fromarray(transparent_background, 'RGBA')

# 旋转90度
rotated_image = result_image.rotate(0, expand=True)

# 保存和显示变换后的图片
rotated_image.save('perspective_transformed_rotated.png')
rotated_image.show()
