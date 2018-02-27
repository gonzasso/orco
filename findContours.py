import argparse
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def get_first_contour(cap):

    fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=False)
    # fgbg = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=200, detectShadows=False)
    count = 0

    kernel = np.ones((5,5),np.uint8)

    while True:
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame, learningRate=0.005)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(opening,100,200)


        # plt.subplot(121),plt.imshow(fgmask,cmap = 'gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        cv2.imshow('fgmask',fgmask)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('closing',fgmask)

        _, cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts)
        for c in cnts:
            # if the contour is too small, ignore it
            area = cv2.contourArea(c)
         #   area = c.shape[0] * c.shape[1]

            if (area < 2000.0 or area > 90000.0):
                continue

            print(area)
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            return x, y, w, h, area

            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # crop_img = frame[y:y + h, x:x + w]

            #cv2.imshow('frame', frame)