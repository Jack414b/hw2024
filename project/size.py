from PIL import Image
import os

# 设置包含原始照片的文件夹路径
input_folder = 'G:\code\dip\hw04\picture'
# 设置调整大小后照片的保存文件夹路径
output_folder = 'G:\code\dip\hw04\picture'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取第一张图片的尺寸
first_image_path = None
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        first_image_path = os.path.join(input_folder, filename)
        break

if first_image_path is None:
    print("没有找到图片文件。")
else:
    with Image.open(first_image_path) as first_image:
        target_size = first_image.size

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            # 确保不是第一张图片
            if image_path != first_image_path:
                with Image.open(image_path) as img:
                    # 调整图片大小
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    # 保存调整大小后的图片
                    img_resized.save(os.path.join(output_folder, filename))

print("所有照片的大小调整完成！")