import imutils
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# combine background & foreground images into out.mp4
# ffmpeg -pattern_type glob -i '*.jpeg' out.mp4
os.chdir('./objects/bg')
cap = cv2.VideoCapture("out.mp4")
# pBackSub = cv2.createBackgroundSubtractorMOG2()
pBackSub = cv2.createBackgroundSubtractorKNN()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgMask = pBackSub.apply(frame)

    # cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)
    if cv2.waitKey(10) == 27:
        break

# cv2.imwrite('mask.jpg', fgMask)
cap.release()
cv2.destroyAllWindows()

plt.imshow(fgMask, cmap='gray')

# 1.Erosion
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(fgMask, kernel, iterations=1)
plt.imshow(erosion, cmap='gray')

# 2.Closing
closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')

# 3.Dilation
dilation = cv2.dilate(fgMask, kernel, iterations=1)
plt.imshow(dilation, cmap='gray')

# 4.Otsu's thresholding
ret, threshold = cv2.threshold(
    fgMask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(threshold, cmap='gray')

'''
background, foreground
'''
# Load image, create mask, and draw white circle on mask
mask = threshold
image = cv2.imread('99.jpeg', 0)

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask < 100] = 255  # Optional
result[mask > 100] = 0  # Optional
plt.imshow(result, cmap='gray')

'''
combined threshold
'''
# Closing
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')

# Erosion
kernel = np.ones((20, 20), np.uint8)
erosion = cv2.erode(closing, kernel, iterations=1)

# Mask input image with binary mask
result = cv2.bitwise_and(image, erosion)
plt.imshow(result, cmap='gray')

cv2.imshow('result', result)
cv2.waitKey(0)
