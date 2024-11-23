from PIL import Image

# 加载小块图片
pieces = []
for i in range(3):
    for j in range(3):
        piece = Image.open(f'small_piece_{i+1}_{j+1}.png')
        pieces.append(piece)

# 获取小块的尺寸
small_width, small_height = pieces[0].size

# 创建新的大图，计算边框的大小
border_size = 5  # 边框大小
new_width = (small_width + border_size) * 3 + border_size
new_height = (small_height + border_size) * 3 + border_size

# 创建一个新的空白图像，背景为白色
new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))

# 将小块放入新的图像中，并添加黑色边框
for i in range(3):
    for j in range(3):
        piece = pieces[i * 3 + j]
        x = j * (small_width + border_size) + border_size
        y = i * (small_height + border_size) + border_size
        
        # 将小块放入新图像
        new_image.paste(piece, (x, y))
        
        # 绘制黑色边框
        for k in range(border_size):
            # 上边框
            new_image.putpixel((x + k, y - 1), (0, 0, 0))
            # 下边框
            new_image.putpixel((x + k, y + small_height), (0, 0, 0))
            # 左边框
            new_image.putpixel((x - 1, y + k), (0, 0, 0))
            # 右边框
            new_image.putpixel((x + small_width, y + k), (0, 0, 0))

# 保存拼接后的大图
new_image.save('stitched_image_with_border.png')

# 显示拼接后的图像
new_image.show()
