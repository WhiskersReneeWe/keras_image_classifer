# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:09:24 2019

@author: Renee Liu
"""

import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3, preprocess_input  
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                                 
import matplotlib.pyplot as plt    
import numpy as np

#train_incep.shape[1:] -- (5,5,2048)

def load_model():
    """
    Input -- There is no input required because this funtion builds the model architecture. But,
    The expected input when using this model should be the output of the
    Bottlebneck feature extractor model
    Output -- a classifier layer for the CNN model
    Requried dependencies -- saved weights from the trained model
    """
    breeds = 133
    INCEPTION_model =  Sequential()
    INCEPTION_model.add(GlobalAveragePooling2D(input_shape = (5,5,2048)))
    INCEPTION_model.add(Dropout(0.4))
    INCEPTION_model.add(Dense(1024, activation='relu'))
    INCEPTION_model.add(Dropout(0.5))
    INCEPTION_model.add(Dense(breeds, activation='softmax'))
    
    optimizer = RMSprop(lr=0.001, rho=0.9)
      
    INCEPTION_model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
    
    INCEPTION_model.load_weights('saved_models/weights.best.INCEPTIONV3.hdf5')
    return INCEPTION_model