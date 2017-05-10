import argparse
import numpy as np
import cv2
import meanShift
import findContours

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

x, y, width, height, area = findContours.get_first_contour(cap)
#x, y, width, height = 0, 0, 200, 200
print(x, y, width, height, area)

meanShift.start_tracking(cap, x, y, width, height)