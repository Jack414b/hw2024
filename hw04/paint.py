import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = cv2.imread('cream.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 

plt.figure()
plt.imshow(input_image)
plt.title('Click on the four corners of the desk')
plt.axis('off')
plt.ion() 
plt.show()

print('Click on the four corners of the desk in clockwise order')
corners = plt.ginput(4) 
plt.close()

x, y = zip(*corners)
input_corners = np.array([x, y], dtype='float32').T

output_corners = np.array([[0, 0], [157, 0], [157, 184], [0, 184]], dtype='float32')

matrix = cv2.getPerspectiveTransform(input_corners, output_corners)

output_size = (157, 184)

warped_image = cv2.warpPerspective(input_image, matrix, output_size)

plt.figure()
plt.imshow(warped_image)
plt.title('Projected Image')
plt.axis('off')
plt.show()

cv2.imwrite('reprojected_image1.jpg', cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))



