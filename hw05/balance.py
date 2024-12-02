import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_white_balance(image, gray_card_region):
    
    gray_card_mean = np.mean(gray_card_region, axis=(0, 1))
    image_mean = np.mean(image, axis=(0, 1))
    scale = gray_card_mean / image_mean
    
    balanced_image = image * scale
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    
    return balanced_image

gray_card_image = cv2.imread('figure1.png')
input_image = cv2.imread('figure2.png')

roi = cv2.selectROI("Select Gray Card Region", gray_card_image, fromCenter=False, showCrosshair=True)
x1, y1, w, h = roi
gray_card_region = gray_card_image[y1:y1+h, x1:x1+w]

gray_card_corrected = correct_white_balance(input_image, gray_card_region)

cv2.destroyWindow("Select Gray Card Region")

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(gray_card_image, cv2.COLOR_BGR2RGB))
plt.title('Gray Card Image (Figure 1)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(gray_card_region, cv2.COLOR_BGR2RGB))
plt.title('Selected Gray Card Region')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image (Figure 2)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(gray_card_corrected, cv2.COLOR_BGR2RGB))
plt.title('Corrected Image using Gray Card (Figure 2)')
plt.axis('off')

plt.tight_layout()
plt.show()

