import os

# 指定图片文件夹路径
folder_path = 'picture'  # 替换为您的文件夹路径

# 获取文件夹下的所有文件
files = os.listdir(folder_path)

# 过滤出图片文件（可以根据需要添加更多扩展名）
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

# 按文件名排序
image_files.sort()

# 重命名文件
for index, filename in enumerate(image_files, start=1):
    # 获取文件的扩展名
    file_extension = os.path.splitext(filename)[1]
    # 新文件名
    new_filename = f'image_{index}{file_extension}'
    # 完整的旧和新文件路径
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)
    
    # 重命名文件
    os.rename(old_file, new_file)
    print(f'Renamed: {filename} to {new_filename}')

print('重命名完成！')
