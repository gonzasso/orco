from __future__ import print_function

import glob

import numpy as np
import cv2
import helpers
import nms
import svmlight
import timeit
import random
import os.path

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
    # TODO chequear limites
    for i in range(0, neg_count):
        labels[i] = -1
    for i in range(neg_count, pos_count + neg_count):
        labels[i] = 1

    np.array(descriptors).flatten()
    descriptors = descriptors.reshape((pos_count + neg_count, hog_size))
    return descriptors, labels


# train SVM for the first time
def trainInitialSVM():
    start_time = timeit.default_timer()
    print ("Calculating hog for training data")
    train_data, labels = getTrainData(hog, ".\/res\img_inria_pos\*", ".\/res\img_inria_neg\*")
    mean_input = np.mean(train_data, axis=0).reshape(1, -1)
    elapsed = timeit.default_timer() - start_time
    print (elapsed)
    # print(mean_input.shape)
    print ("Calculating PCA")
    mean, eigenvectors = cv2.PCACompute(train_data, mean_input, cv2.PCA_DATA_AS_ROW, 512)
    np.save("./hog_descriptors", train_data)
    np.save("./labels", labels)
    np.save("./pca_eigenvectors", eigenvectors)
    np.save("./pca_mean", mean)
    # train_data = np.load("./hog_descriptors.npy")
    # labels = np.load("./labels.npy")
    # eigenvectors = np.load("./pca_eigenvectors.npy")
    # mean = np.load("./pca_mean.npy")
    projection = cv2.PCAProject(train_data, mean, eigenvectors)
    elapsed = timeit.default_timer() - elapsed
    print (elapsed)
    np.save('./pca_projection', projection)
    print ("Training SVM")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.5)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    svm.save("svm.xml")
    elapsed = timeit.default_timer() - elapsed
    print(elapsed)


# def trainInitialSVM():
#     train_data, labels = getTrainData(hog, ".\/train\/pos\*.jpg", ".\/train\/neg\*.jpg")
#     mean_input = np.mean(train_dat    a, axis=0).reshape(1, -1)
#     # print(mean_input.shape)
#     mean, eigenvectors = cv2.PCACompute(train_data, mean_input, cv2.PCA_DATA_AS_ROW, 512)
#     np.save("./hog_descriptors", train_data)
#     np.save("./labels", labels)
#     np.save("./pca_eigenvectors", eigenvectors)
#     np.save("./pca_mean", mean)
#     # train_data = np.load("./hog_descriptors.npy")
#     # labels = np.load("./labels.npy")
#     # eigenvectors = np.load("./pca_eigenvectors.npy")
#     # mean = np.load("./pca_mean.npy")
#     projection = cv2.PCAProject(train_data, mean, eigenvectors)
#     np.save('./pca_projection', projection)
#     # projection = np.load('./pca_projection.npy')
#     train = []
#     for i in range(0, len(labels)-1):
#         feature = []
#         descriptor = projection[i]
#         for j in range(0, len(descriptor)-1):
#             feature.append((j+1, descriptor[j]))
#         train.append((labels[i], feature))
#
#     model = svmlight.learn(train, type='classification', kernel='linear')
#     svmlight.write_model(model, 'my_model.dat')

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


# get negative images from inria dataset
def getNegImages():
    extensions = ("*.jpg", "*.png")
    path = "C:\Users\gonza\Downloads\INRIAPerson\Train\/neg\/"
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + extension))
    for image_file in files:
        for i in range(0, 9):
            image = cv2.imread(image_file)
            height, width, _ = image.shape
            x = random.randint(0, width - 64)
            y = random.randint(0, height - 128)
            crop_img = image[y:y + 128, x:x + 64]
            cv2.imwrite(".\img_inria\/crop%d_%s" % (i, os.path.basename(image_file)), crop_img)


# get positive images from inria dataset
def getPosImg():
    extensions = ("*.jpg", "*.png")
    path = "C:\Users\gonza\Downloads\INRIAPerson\96X160H96\Train\pos\/"
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + extension))
    for image_file in files:
        image = cv2.imread(image_file)
        height, width, _ = image.shape
        crop_img = image[16:height - 16, 16:width - 16]
        cv2.imwrite(".\img_inria_pos\/crop_%s" % os.path.basename(image_file), crop_img)


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
    # mean = np.load("./pca_mean.npy")
    eigenvectors = np.load("./pca_eigenvectors.npy")
    mean = np.mean(img_descriptor, axis=0).reshape(1, -1)
    return cv2.PCAProject(img_descriptor, mean, eigenvectors)

