# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:18:59 2019
Dog Face and human face detecter
@author: Renee Liu
"""

import cv2                
import matplotlib.pyplot as plt                        
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
import numpy as np


def ResNet50_predict_labels(img):
    '''
    input - any image, a specified targeted size
    output - predict what this image's label is based on the pre-defined categories from ImageNet
    '''
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    img = preprocess_input(img)
    ResNet50_model = ResNet50(weights='imagenet')
    return np.argmax(ResNet50_model.predict(img))


# returns "True" if face is detected in image stored at img_path
def face_detecting(img):
    '''
    input - any image, a specified targeted size
    output - Return true if a human face is detected from the image input
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detecting(img):
    '''
    input -- the preprocessed image for Keras Model
    output -- return true if a dog face is detected based on the existing labels from ImageNet
    '''
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 

