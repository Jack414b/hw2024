import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from scipy.fft import fft2, ifft2, fftshift

# 均值滤波器
def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return Image.fromarray(convolve(np.array(image), kernel, mode='reflect'))

# 高斯滤波器
def gaussian_filter_custom(image, sigma):
    return Image.fromarray(gaussian_filter(np.array(image), sigma=sigma))

# 拉普拉斯滤波器
def laplacian_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    return Image.fromarray(convolve(np.array(image), kernel, mode='reflect'))

def sharpening_filter(image, alpha=1.5):
    """锐化滤波器，增加了权重控制"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5 * alpha, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened_image = convolve(np.array(image), kernel, mode='reflect')
    sharpened_image = np.clip(sharpened_image, 0, 255)  # 确保像素值在0-255之间
    return Image.fromarray(sharpened_image.astype(np.uint8))

# 频域滤波
def frequency_domain_filter(image):
    image_array = np.array(image)
    rows, cols, channels = image_array.shape
    crow, ccol = rows // 2, cols // 2
    radius = 30  # 低通滤波器的半径

    low_pass_img = np.zeros_like(image_array)
    high_pass_img = np.zeros_like(image_array)

    for channel in range(channels):
        f_transform = fft2(image_array[:, :, channel])
        f_transform_shifted = fftshift(f_transform)

        # 理想低通滤波器
        low_pass_filter = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
                    low_pass_filter[i, j] = 1

        # 应用低通滤波器
        filtered_low = f_transform_shifted * low_pass_filter
        img_low_pass = np.abs(ifft2(filtered_low))

        # 理想高通滤波器
        high_pass_filter = 1 - low_pass_filter
        filtered_high = f_transform_shifted * high_pass_filter
        img_high_pass = np.abs(ifft2(filtered_high))

        low_pass_img[:, :, channel] = np.clip(img_low_pass, 0, 255)
        high_pass_img[:, :, channel] = np.clip(img_high_pass, 0, 255)

    return Image.fromarray(low_pass_img.astype(np.uint8)), Image.fromarray(high_pass_img.astype(np.uint8))

# 处理多张图像的函数
def process_images(image_paths):
    output_dir = 'filtered_output_dir'
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    for image_path in image_paths:
        # 读取图像
        image = Image.open(image_path) 
        
        # 分离RGB通道
        r, g, b = image.split()

        # 应用滤波器
        mean_img_r = mean_filter(r, 3)
        mean_img_g = mean_filter(g, 3)
        mean_img_b = mean_filter(b, 3)
        mean_img = Image.merge('RGB', (mean_img_r, mean_img_g, mean_img_b))

        gaussian_img_r = gaussian_filter_custom(r, sigma=1)
        gaussian_img_g = gaussian_filter_custom(g, sigma=1)
        gaussian_img_b = gaussian_filter_custom(b, sigma=1)
        gaussian_img = Image.merge('RGB', (gaussian_img_r, gaussian_img_g, gaussian_img_b))

        laplacian_img_r = laplacian_filter(r)
        laplacian_img_g = laplacian_filter(g)
        laplacian_img_b = laplacian_filter(b)
        laplacian_img = Image.merge('RGB', (laplacian_img_r, laplacian_img_g, laplacian_img_b))

        sharpening_img_r = sharpening_filter(r)
        sharpening_img_g = sharpening_filter(g)
        sharpening_img_b = sharpening_filter(b)
        sharpening_img = Image.merge('RGB', (sharpening_img_r, sharpening_img_g, sharpening_img_b))

        low_pass_img, high_pass_img = frequency_domain_filter(image)

        # 保存处理后的图像
        base_name = os.path.basename(image_path).split('.')[0]
        mean_img.save(os.path.join(output_dir, f'{base_name}_mean_filtered.png'))
        gaussian_img.save(os.path.join(output_dir, f'{base_name}_gaussian_filtered.png'))
        laplacian_img.save(os.path.join(output_dir, f'{base_name}_laplacian_filtered.png'))
        sharpening_img.save(os.path.join(output_dir, f'{base_name}_sharpened.png'))
        low_pass_img.save(os.path.join(output_dir, f'{base_name}_low_pass_filtered.png'))
        high_pass_img.save(os.path.join(output_dir, f'{base_name}_high_pass_filtered.png'))

        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1), plt.imshow(image), plt.title('Original Image')
        plt.subplot(2, 3, 2), plt.imshow(mean_img), plt.title('Mean Filtered')
        plt.subplot(2, 3, 3), plt.imshow(gaussian_img), plt.title('Gaussian Filtered')
        plt.subplot(2, 3, 4), plt.imshow(laplacian_img), plt.title('Laplacian Filtered')
        plt.subplot(2, 3, 5), plt.imshow(sharpening_img), plt.title('Sharpened')
        plt.subplot(2, 3, 6), plt.imshow(low_pass_img), plt.title('Low Pass Filtered')
        plt.show()

# 图像文件
selected_images = ['output_dir/gaussian_noisy_IMG_1748.JPG', 
                   'output_dir/sp_noisy_IMG_1748.JPG', 
                   'output_dir/poisson_nOISY_IMG_1748.JPG'] 
process_images(selected_images)
