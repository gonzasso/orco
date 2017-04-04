import argparse
import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=100, detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=200, detectShadows=False)

while True:
    ret, frame = cap.read()

    # foreground_detection = frame.copy()

    fgmask = fgbg.apply(frame, learningRate=0.005)

    fgmask = cv2.dilate(fgmask, None, iterations=2)

    foreground_img = cv2.bitwise_and(frame, frame, mask = fgmask)

    foundLocations, _ = hog.detectMultiScale(fgmask, 0, (8, 8), (32, 32), 1.05, 2)

    # find contours and hierarchy in frame
    _, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    # fill contours
    # cv2.drawContours(fgmask, cnts, -1, (255, 255, 255), thickness=cv2.FILLED, hierarchy=hierarchy, maxLevel=2)

    # define params for blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 750
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByColor = True
    params.blobColor = 255
    # create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # find blobs in frame
    blobs = detector.detect(fgmask)

    im_with_keypoints = frame.copy()
    # # loop over the contours
    # for c in cnts:
    #     # if the contour is too small, ignore it
    #     if cv2.contourArea(c) < args["min_area"]:
    #         continue
    #
    #     # compute the bounding box for the contour, draw it on the frame,
    #     # and update the text
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     # draw circles for the blobs found
    #     # im_with_keypoints = cv2.drawKeypoints(frame, blobs, np.array([]), (0, 0, 255),
    #     #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     # cv2.imshow('frame', im_with_keypoints)

    for person in foundLocations:
        (x, y, w, h) = person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('foreground', foreground_img)
    cv2.imshow('frame', frame)
    cv2.imshow('bgs', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
