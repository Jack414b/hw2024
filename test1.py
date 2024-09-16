from PIL import Image
import matplotlib.pyplot as plt

#导入图片
image_path = 'jgy.jpg'
image = Image.open(image_path)

plt.imshow(image)
plt.axis('off')
plt.show()


