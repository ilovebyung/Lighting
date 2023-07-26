import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

'''
get background from a video 
'''
file = 'D:/source/object_tracking/highway.mp4'
video = cv2.VideoCapture(file)

FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)


# creating an array of frames from frames chosen above
frames = []
for frameOI in FOI:
    video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
    ret, frame = video.read()
    frames.append(frame)

# calculate the average
backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
cv2.imwrite("bg.jpg", backgroundFrame)
cv2.imshow('', backgroundFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
get background from files
'''
images = []
files = glob.glob('*.jpg')
for file in files:
    image = cv2.imread(file)
    images.append(image)

# calculate the average
backgroundFrame = np.median(images, axis=0).astype(dtype=np.uint8)
cv2.imwrite("bg.jpg", backgroundFrame)
image = plt.imread('bg.jpg')
plt.imshow(image)

