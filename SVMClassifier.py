from __future__ import print_function

import glob

import numpy as np
import cv2
import helpers
import nms

hog = cv2.HOGDescriptor()


# get list of HOG descriptors and labels for positive and negative images
def getTrainData(hogDesc, positiveSamplesPath, negSamplesPath):
    pos_count = neg_count = 0
    hog_size = 0
    descriptors = None
    for neg_img in glob.glob(negSamplesPath):
        img = cv2.imread(neg_img)
        neg_count += 1
        descriptor = hogDesc.compute(img)
        if descriptors is None:
            descriptors = descriptor
            hog_size = descriptor.shape[0]
        else:
            descriptors = np.vstack((descriptors, descriptor))

    for pos_img in glob.glob(positiveSamplesPath):
        pos_count += 1
        img = cv2.imread(pos_img)
        descriptor = hogDesc.compute(img)
        descriptors = np.vstack((descriptors, descriptor))

    labels = np.empty((pos_count + neg_count, 1), np.int32)
    np.array(labels).fill(1)
    for i in range(0, neg_count - 1):
        labels[i] = -1

    np.array(descriptors).flatten()
    descriptors = descriptors.reshape((pos_count + neg_count, hog_size))
    return descriptors, labels


# train SVM for the first time
def trainInitialSVM():
    train_data, labels = getTrainData(hog, ".\/train\pos\*.jpg", ".\/train\/neg\*.jpg")
    mean_input = np.mean(train_data, axis=0).reshape(1, -1)
    # print(mean_input.shape)
    mean, eigenvectors = cv2.PCACompute(train_data, mean_input, cv2.PCA_DATA_AS_ROW, 512)
    # np.save("./hog_descriptors", train_data)
    # np.save("./labels", labels)
    np.save("./pca_eigenvectors", eigenvectors)
    # np.save("./pca_mean", mean)
    # train_data = np.load("./hog_descriptors.npy")
    # labels = np.load("./labels.npy")
    # eigenvectors = np.load("./pca_eigenvectors.npy")
    # mean = np.load("./pca_mean.npy")
    projection = cv2.PCAProject(train_data, mean, eigenvectors)
    print(projection)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.5)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    svm.save("svm.xml")


def createTrainImages():
    count = 0
    for file in glob.glob(".\/train\/pos\*.png"):
        img = cv2.imread(file)
        resized = cv2.resize(img, (64, 128))
        cv2.imwrite(".\/train\positive\/img%d.jpg" % count, resized)
        count += 1

    count = 0
    for neg_img in glob.glob(".\/train\/neg\*.png"):
        negative = cv2.imread(neg_img)
        for (x, y, window) in helpers.sliding_window(negative, stepSize=64, windowSize=(64, 128)):
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue
            cv2.imwrite(".\/train\/negative\/img%d.jpg" % count, window)
            count += 1


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


def testImg():
    svm = cv2.ml.SVM_load(".\svm.xml")
    for file in glob.glob(".\/test\pos\*.png"):
        img = cv2.imread(file)
        boundingList = None
        for (x, y, window) in helpers.sliding_window(img, stepSize=32, windowSize=(64, 128)):
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue

            winDescriptor = hog.compute(window)
            rows, cols = winDescriptor.shape
            winDescriptor = winDescriptor.reshape((cols, rows))
            prediction = svm.predict(winDescriptor)
            if prediction[1] == 0:
                boundingBox = (x, y, x + 64, y + 128)
                if boundingList is None:
                    boundingList = np.array([boundingBox])
                else:
                    boundingList = np.vstack((boundingList, [boundingBox]))
            else:
                continue
            pick = nms.non_max_suppression_slow(boundingList, 0.3)
            x, y, w, h = pick[0]
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            cv2.imshow("image", img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


def projectData(img_descriptor):
    # mean = np.load("./pca_mean")
    eigenvectors = np.load("./pca_eigenvectors.npy")
    mean = np.mean(img_descriptor, axis=0).reshape(1, -1)
    return cv2.PCAProject(img_descriptor, mean, eigenvectors)


trainInitialSVM()
