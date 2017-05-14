import imutils
import cv2
import numpy as np


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # loop over the pyramid
    while True:
        # compute new dimensions of image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield next image in pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def normalize_image(image):
    image = np.float32(image)

    b, g, r = cv2.split(image)
    f_intensity = (b+g+r) / 3.0
    f_intensity = np.uint8(f_intensity)

    g_normalized_img = np.divide(r, f_intensity)

    g_intensity = np.multiply(g_normalized_img, 255.0)
    g_intensity = np.uint8(g_intensity)
    # g_intensity = cv2.convertScaleAbs(g_normalized_img, alpha=0, beta=255.0)

    return f_intensity

