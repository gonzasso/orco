import argparse
import cv2
import math
import time
# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

vidcap = cv2.VideoCapture(args["video"])

success,image = vidcap.read()
count = 0
frameRate = vidcap.get(5) #frame rate
while success:
    frameId = vidcap.get(1) #current frame number
    success, frame = vidcap.read()
    if frameId % math.floor(frameRate) == 0:
        cv2.imwrite("frame%d.jpg" % count, frame)
        count += 1
