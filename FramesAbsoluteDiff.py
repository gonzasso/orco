import cv2
import numpy as np
import math

backgroundImage = cv2.imread("C:\Users\gonza\PycharmProjects\orco\/frame0")
compareImage = cv2.imread("C:\Users\gonza\PycharmProjects\orco\/frame12")
diffImage = np.zeros((255, 255, 1), dtype="uint8")
cv2.absdiff(backgroundImage, compareImage, diffImage)
foregroundMask = np.zeros(diffImage.shape, dtype="uint8")

threshold = 30.0

for (x, y), value in np.ndenumerate(diffImage[:, :, 0]):
    pix = diffImage[x, y, :]

    dist = math.sqrt(pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2])

    if dist > threshold:
        foregroundMask[x, y, :] = 255
