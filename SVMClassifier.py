from __future__ import print_function

import glob

import numpy as np
import cv2
import helpers

hog = cv2.HOGDescriptor()


# get list of HOG descriptors and labels for positive and negative images
def getTrainData(hogDesc, positiveSamplesPath, negSamplesPath):
    pos_count = neg_count = 0
    hog_size = 0
    descriptors = None
    for pos_img in glob.glob(negSamplesPath):
        img = cv2.imread(pos_img)
        neg_count += 1
        descriptor = hogDesc.compute(img)
        if descriptors is None:
            descriptors = descriptor
            hog_size = descriptor.shape[0]
        else:
            descriptors = np.vstack((descriptors, descriptor))

    for neg_img in glob.glob(positiveSamplesPath):
        pos_count += 1
        img = cv2.imread(neg_img)
        descriptor = hogDesc.compute(img)
        descriptors = np.vstack((descriptors, descriptor))

    labels = np.empty((pos_count + neg_count, 1), np.int32)
    np.array(labels).fill(1)
    for i in range(0, neg_count - 1):
        labels[i] = -1

    np.array(descriptors).flatten()
    print(labels.shape)
    descriptors = descriptors.reshape((pos_count + neg_count, hog_size))
    print(descriptors.shape)
    return descriptors, labels


# train SVM for the first time
def trainInitialSVM():
    train_data, labels = getTrainData(hog, ".\img_positive\*.jpg", ".\img_neg\*.jpg")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1.0)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    svm.save("svm.xml")


# apply Hard Negative Mining to improve the SVM
def hardNegativeMining():
    svm = cv2.ml.SVM_load(".\svm.xml")

    (winW, winH) = (64, 128)
    # for each scale of every image in the negative test dataset apply the sliding window technique
    # get HOG descriptor of every window and use SVM to predict results
    for image_file in glob.glob(".\/test\/neg\*.png"):
        image = cv2.imread(image_file)
        for resized in helpers.pyramid(image, scale=1.5):
            for (x, y, window) in helpers.sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                winDescriptor = hog.compute(window)
                rows, cols = winDescriptor.shape
                winDescriptor = winDescriptor.reshape((cols, rows))
                prediction = svm.predict(winDescriptor)
                print(prediction)

    # TODO save predictions of false positives ordered by their probabilities and retrain the model
