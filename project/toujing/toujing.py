from PIL import Image, ImageDraw

# 定义微透镜阵列的参数
lens_diameter = 10  # 每个透镜的直径
rows = 50           # 透镜的行数
cols = 100          # 透镜的列数

# 创建一个白色背景的图像，长宽比为2:1
image_width = 1000  # 图像宽度
image_height = 500   # 图像高度
image = Image.new("RGB", (image_width, image_height), "white")
draw = ImageDraw.Draw(image)

# 计算每个微透镜的实际位置
for row in range(rows):
    for col in range(cols):
        # 计算透镜的左上角和右下角坐标
        x0 = col * lens_diameter
        y0 = row * lens_diameter
        x1 = x0 + lens_diameter
        y1 = y0 + lens_diameter
        
        # 绘制圆形（微透镜）
        draw.ellipse([x0, y0, x1, y1], fill=(135, 206, 250), outline="black")  # 浅蓝色

        # 添加弧面反光效果
        for i in range(5):  # 创建渐变效果
            reflection_color = (255, 255, 255, 255 - i * 50)  # 渐变白色
            draw.ellipse([x0 + 2, y0 + 2 + i, x1 - 2, y1 - (lens_diameter / 2) + i], fill=reflection_color)

# 保存和显示图像
image.save('microlens_array_circles_2_to_1.png')
image.show()

