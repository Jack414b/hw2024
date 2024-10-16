import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 获取当前工作目录
image_folder = os.getcwd()  # 当前文件夹
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.JPG', '.jpeg', '.jpg'))]

# 选择适合滤波的图像
selected_images = image_files[:3]

# 添加噪声函数
def add_gaussian_noise(image, mean=0, sigma=200):
    """添加高斯噪声"""
    gauss = np.random.normal(mean, sigma, image.size)
    noisy_image = np.clip(np.array(image) + gauss.reshape(image.size[1], image.size[0], -1), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1):
    """添加盐和胡椒噪声"""
    noisy_image = np.array(image)
    total_pixels = noisy_image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # 添加胡椒噪声
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return Image.fromarray(noisy_image)

def add_poisson_noise(image, scale=0.5):
    """添加泊松噪声"""
    noisy_image = np.array(image).astype(np.float32)  # 转换为浮点型以避免溢出
    noise = np.random.poisson(noisy_image * scale)  # 使用缩放因子控制噪声强度
    noisy_image = np.clip(noisy_image + noise, 0, 255)  # 确保像素值在 0 到 255 之间
    return Image.fromarray(noisy_image.astype(np.uint8))  # 转换为 uint8 类型并返回图

# 显示原图像和添加噪声后的图像
plt.figure(figsize=(15, 10))
for i, image_file in enumerate(selected_images):
    img_path = os.path.join(image_folder, image_file)
    image = Image.open(img_path)

    # 添加噪声
    gaussian_noisy = add_gaussian_noise(image)
    sp_noisy = add_salt_and_pepper_noise(image)
    poisson_noisy = add_poisson_noise(image)
    
    #保存图片
    output_dir='output_dir'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(image_file)
    gaussian_noisy.save(os.path.join(output_dir, f'gaussian_noisy_{base_name}'))
    sp_noisy.save(os.path.join(output_dir, f'sp_noisy_{base_name}'))
    poisson_noisy.save(os.path.join(output_dir, f'poisson_noisy_{base_name}'))

    # 显示原图像和噪声图像
    plt.subplot(4, 4, i * 4 + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Original: {image_file}')

    plt.subplot(4, 4, i * 4 + 2)
    plt.imshow(gaussian_noisy)
    plt.axis('off')
    plt.title('Gaussian Noise')

    plt.subplot(4, 4, i * 4 + 3)
    plt.imshow(sp_noisy)
    plt.axis('off')
    plt.title('Salt & Pepper Noise')
    
    plt.subplot(4, 4, i * 4 + 4)
    plt.imshow(poisson_noisy)
    plt.axis('off')
    plt.title('Poisson Noise')

plt.tight_layout()
plt.show()
