from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = Image.open('jgy.jpg').convert('L')
image_array = np.array(image)

# 计算灰度直方图
histogram, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256])

# 计算累计分布函数（CDF）
cdf = histogram.cumsum()
cdf_normalized = cdf * histogram.max() / cdf.max()  # 归一化CDF

# 根据CDF进行像素强度映射
cdf_mapped = np.ma.masked_equal(cdf, 0)  # 避免除以零
cdf_mapped = (cdf_mapped - cdf_mapped.min()) * 255 / (cdf_mapped.max() - cdf_mapped.min())  # 归一化到[0, 255]
cdf_mapped = np.round(cdf_mapped).astype(np.uint8)

# 映射图像
equalized_image_array = cdf_mapped[image_array]
equalized_image = Image.fromarray(equalized_image_array)

# 显示原始图像与均衡化后的图像及其直方图
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# 均衡化后的图像
plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

# 原始图像直方图
plt.subplot(2, 2, 3)
plt.title('Original Histogram')
plt.hist(image_array.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlim([0, 256])

# 均衡化后的图像直方图
plt.subplot(2, 2, 4)
plt.title('Equalized Histogram')
plt.hist(equalized_image_array.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
