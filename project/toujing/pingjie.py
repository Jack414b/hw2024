from PIL import Image
import matplotlib.pyplot as plt

# 加载图片
image_path = 'chessboard.jpeg'  # 替换为你的图片路径
image = Image.open(image_path)

# 获取图片的宽度和高度
width, height = image.size

# 计算每一小块的宽度和高度
small_width = width // 3
small_height = height // 3

# 分割图片并保存
for i in range(3):
    for j in range(3):
        # 计算每一小块的位置
        left = j * small_width
        upper = i * small_height
        right = left + small_width
        lower = upper + small_height
        
        # 裁剪小块
        small_piece = image.crop((left, upper, right, lower))
        
        # 保存小块
        small_piece.save(f'small_piece_{i+1}_{j+1}.png')

        # 可选：显示小块
        plt.imshow(small_piece)
        plt.title(f'小块 ({i+1},{j+1})')
        plt.axis('off')
        plt.show()
