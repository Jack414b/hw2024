import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 相机1的旋转矩阵和平移向量
R1 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
t1 = np.array([0, 0, 0])

# 相机2的旋转矩阵和平移向量
R2 = np.array([[-0.1182, -0.7956, 0.5941],
               [-0.9930, 0.0900, -0.0770],
               [0.0078, -0.5991, 0.8007]])
t2 = np.array([-147.51, 202.1139, 90.8602])

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制相机1的位置
ax.quiver(t1[0], t1[1], t1[2],
          R1[0,0], R1[0,1], R1[0,2],
          color='r', length=50, normalize=True)
ax.quiver(t1[0], t1[1], t1[2],
          R1[1,0], R1[1,1], R1[1,2],
          color='g', length=50, normalize=True)
ax.quiver(t1[0], t1[1], t1[2],
          R1[2,0], R1[2,1], R1[2,2],
          color='b', length=50, normalize=True)

# 绘制相机2的位置
ax.quiver(t2[0], t2[1], t2[2],
          R2[0,0], R2[0,1], R2[0,2],
          color='r', length=50, normalize=True)
ax.quiver(t2[0], t2[1], t2[2],
          R2[1,0], R2[1,1], R2[1,2],
          color='g', length=50, normalize=True)
ax.quiver(t2[0], t2[1], t2[2],
          R2[2,0], R2[2,1], R2[2,2],
          color='b', length=50, normalize=True)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-300, 300])
ax.set_ylim([-300, 300])
ax.set_zlim([-300, 300])
ax.set_aspect('auto')
ax.set_title('Camera Poses in 3D Space')

# 显示图形
plt.show()

