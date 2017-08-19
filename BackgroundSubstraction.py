import argparse
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import SVMClassifier
import helpers

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.ml.SVM_load(".\svm.xml"))

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=200, detectShadows=False)
count = 0

svm = cv2.ml.SVM_load(".\svm.xml")

while True:
    ret, frame = cap.read()

    norm_image = frame.copy()

    fgmask = fgbg.apply(frame, learningRate=0.005)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    foreground_img = cv2.bitwise_and(frame, frame, mask=fgmask)

    # foundLocations, _ = hog.detectMultiScale(foreground_img, 0, (8, 8), (32, 32), 1.05, 2)

    # find contours and hierarchy in frame
    _, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    # fill contours
    # cv2.drawContours(fgmask, cnts, -1, (255, 255, 255), thickness=cv2.FILLED, hierarchy=hierarchy, maxLevel=2)

    # im_with_keypoints = frame.copy()
    # # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        crop_img = frame[y:y + h, x:x + w]

        TARGET_PIXEL_AREA = 100000.0

        ratio = float(crop_img.shape[1]) / float(crop_img.shape[0])
        if ratio >= 1:
            continue

        # new_h = int(np.math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
        # new_w = int((new_h * ratio) + 0.5)

        crop_img = cv2.resize(crop_img, (64, 128))

        descriptor = hog.compute(crop_img)
        rows, cols = descriptor.shape
        descriptor = descriptor.reshape((cols, rows))
        descriptor = SVMClassifier.projectData(descriptor)
        prediction = svm.predict(descriptor)
        if prediction[1] == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rows, cols = crop_img.shape[:2]

#         crop_img = cv2.flip(crop_img, 1)
#         rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
#         crop_img_flip = cv2.warpAffine(crop_img, rot_matrix, (cols, rows))
#         cv2.imwrite(".\img\/contour%d.jpg" % count, crop_img)
#         cv2.imwrite(".\img\/contour_duplicate%d.jpg" % count, crop_img_flip)
#         count += 1
#         draw circles for the blobs found
#         im_with_keypoints = cv2.drawKeypoints(frame, blobs, np.array([]), (0, 0, 255),
#                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         cv2.imshow('frame', im_with_keypoints)
#
#     for person in foundLocations:
#         (x, y, w, h) = person
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
    cv2.imshow('foreground', frame)
    cv2.imshow('frame', frame)
    cv2.imshow('bgs', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

