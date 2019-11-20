import cv2
import numpy as np
import itertools
import pandas as pd
import tensorflow as tf
import preprocess
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras


def rescale_segment(segment, size=[28, 28], pad=0):
    '''function for resizing (scaling down) images
    input parameters
    seg : the segment of image (np.array)
    size : out size (list of two integers)F
    output
    scaled down image'''
    if len(segment.shape) == 3:  # Non Binary Image
        import cv2
        # thresholding the image
        ret, segment = cv2.threshold(segment, 127, 255, cv2.THRESH_BINARY)
    m, n = segment.shape
    idx1 = list(range(0, m, (m) // (size[0])))
    idx2 = list(range(0, n, n // (size[1])))
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i, j] = segment[idx1[i] +
                                (m % size[0]) // 2, idx2[j] + (n % size[0]) // 2]
    return out


def extract_segments(img, pad=30, reshape=0, size=[28, 28], area=150, threshold=100,
                     gray=False, dil=True, ker=1):
    '''function to extract individual chacters and digits from an image
    input paramterts
    img : input image (numpy array)
    pad : padding window size around segments (int)
    size : out size (list of two integers)
    reshape : if 1 , output will be scaled down. if 0, no scaling down
    area : Minimum area requirement for connected component detection
    thresh : gray scale to binary threshold value
    gray : if False, the segments returned will be binary, else will be gray scale
    dil : if True, performs dilation on segments, else erosion
    ker : dimesnion of kernel size for dilation / erosion
    Returns
    out : list of each segments (starting from leftmost digit)'''

    import cv2

    # thresholding the image
    # img = img.astype(np.uint8)
    ret, thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Negative tranform gray levels (background becomes black)
    thresh1 = 255 - thresh1
    img = 255 - img

    # connected component labelling
    output = cv2.connectedComponentsWithStats(thresh1, 4)
    final = []
    temp2 = output[2]
    temp2 = temp2[temp2[:, 4] > area]
    temp1 = np.sort(temp2[:, 0])
    kernel = np.ones([ker, ker])

    for i in range(1, temp2.shape[0]):
        cord = np.squeeze(temp2[temp2[:, 0] == temp1[i]])
        #         import pdb; pdb.set_trace()
        #         print(cord)

        if gray == False:
            num = np.pad(thresh1[cord[1]:cord[1] + cord[3],
                                 cord[0]:cord[0] + cord[2]], pad, 'constant')
        else:
            num = np.pad(img[cord[1]:cord[1] + cord[3], cord[0]                             :cord[0] + cord[2]], pad, 'constant')

        if dil:
            num = cv2.dilate(num, kernel, iterations=1)
        else:
            num = cv2.erode(num, kernel, iterations=1)

        if reshape == 1:
            num = rescale_segment(num, size)
        final.append(num / 255)

    return final


"""def load_data(class_labels, data_name, label_name, train=0.85, val=0.15):
    Function to Load data from .npy files and split them into training and validation sets
    Inputs
    class labels : Dictionary of class labels (dict)
    data_name : name of .npy data file, with path (str)
    label_name : name of .npy label file, with path (str)
    train : fraction of samples used in training set (float)
    val : fraction of samples used in training set (float)
    
    data = pd.DataFrame(np.load(data_name))
    labels = pd.DataFrame(np.load(label_name))

    labels = labels.rename(columns={0: 'labels'})

    labels['labels'] = labels['labels'].map(class_labels)
    assert data.shape[0] == labels.shape[0]
    assert isinstance(train, float)
    isinstance(val, float), "train and val must be of type float, not {0} and {1}".format(type(train), type(val))
    assert ((train + val) == 1.0), "train + val must equal 1.0"

    one_hot = pd.get_dummies(labels['labels'])
    sidx = int(data.shape[0] * train)
    _data = {'train': data.iloc[:sidx].as_matrix(), 'val': data.iloc[sidx + 1:].as_matrix()}
    _labels = {'train': one_hot.iloc[:sidx, :].as_matrix(), 'val': one_hot.iloc[sidx + 1:, :].as_matrix()}

    assert (_data['train'].shape[0] == _labels['train'].shape[0])
    assert (_data['val'].shape[0] == _labels['val'].shape[0])
    return _data, _labels"""


def load_models():
    '''Function to Load trained models from given path, assuming the names of each models are known
    Inputs
    path : path to the directory with stored models (str)
    ver : version of models loaded (1 = 20000 MNIST + HASY , 2 = 60000 MNIST + Kaggle), 0 for custom models
    names : names of various models'''
    json_file1 = open('./model/model.json', 'r')
    loaded_model_json1 = json_file1.read()
    json_file1.close()
    CNN_ver1 = model_from_json(loaded_model_json1)
    # load weights into new model
    CNN_ver1.load_weights('./model/model_weights.h5')
    adam1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    CNN_ver1.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=adam1, metrics=['accuracy'])

    #print("Loaded models from disk")
    return CNN_ver1


def predict(temp, label_class, prob=0, CNN=True):
    '''Function for using trained models to make prediction, and return predictions
    Inputs
    temp : image (segment) of digit / symbol
    label_class : dictionary of class labels
    prob : if 1, gives probability of each prediction
    CNN : Trained CNN model'''

    # predictions = []
    temp1 = temp.reshape(temp.shape[0], 28, 28, 1)
    cnn_pred = label_class[np.argmax(CNN.predict(temp1))]
    cnn_pred += '({0:1.3f})'.format(np.max(CNN.predict_proba(temp1,
                                                             verbose=0))) * prob
    # predictions.append(cnn_pred)
    return cnn_pred
