from PIL import Image
import matplotlib.pyplot as plt

#导入图片
image_path = 'jgy.jpg'
image = Image.open(image_path)

#转换灰度图
gray_image = image.convert('L')

#计算灰度直方图
histogram = gray_image.histogram()

plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram, width=1, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim(0, 255)  # 设置横坐标范围
plt.grid(axis='y')

# 显示直方图
plt.show()