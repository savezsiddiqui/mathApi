import cv2
import numpy as np
import itertools
import tensorflow as tf
import preprocess
from preprocess import rescale_segment as rescale_segment
from preprocess import extract_segments as extract_segments


def result():
    img = cv2.imread('./uploads/some_image.png', 0)
    img = cv2.resize(img, (368, 75))
    image = []
    for i in range(1):
        for j in range(1):
            x1 = 370 * i
            x2 = x1 + 350
            y1 = 500 * j
            y2 = y1 + 500
            temp = img[x1:x2, y1:y2]
            kernel = np.ones([3, 3])
            temp = cv2.erode(temp, kernel, iterations=1)
            image.append(temp)

    print(len(image))
    for i in range(len(image)):
        im1 = image[i]
        segments = extract_segments(im1, 30, reshape=1, size=[28, 28],
                                    threshold=40, area=100, ker=1, gray=True)


    class_labels = {str(x): x for x in range(10)}
    class_labels.update({'+': 10, '*': 11, '-': 12})
    label_class = dict(zip(class_labels.values(), class_labels.keys()))

    #load model
    cnn_ver1 = preprocess.load_models()

    cnn_pred = ''
    for i in range(len(segments)):
        temp = segments[i]
        temp = temp.reshape(1, -1)
        pred = preprocess.predict(temp, label_class, 0, cnn_ver1)
        cnn_pred += pred

    return cnn_pred
