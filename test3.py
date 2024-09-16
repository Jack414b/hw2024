from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = Image.open('jgy.jpg').convert('L')  # 转换为灰度图

# 定义自定义曲线（简单的非线性变换)
x_curve = np.arange(256)
y_curve = np.clip((x_curve ** 1.2) / 255 * 255, 0, 255).astype(np.uint8)  # 自定义曲线

# 创建查找表
lookup_table = list(y_curve)

# 应用查找表进行点变换
transformed_image = image.point(lookup_table)

# 显示原图和处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(transformed_image, cmap='gray')
plt.axis('off')

plt.show()
