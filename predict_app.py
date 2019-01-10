# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:24:09 2019

@author: shiyan
"""

import base64
import numpy as np
import io
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
from tqdm import tqdm
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline  


from flask import request, jsonify, Flask

app = Flask(__name__)

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def ResNet50_predict_labels(img):
    # returns prediction vector for image located at img_path
    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_image(img, target_size = (224, 224))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img):
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img):
    img = preprocess_image(img, target_size = (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def get_model():
    breeds = 133
    INCEPTION_model =  Sequential()
    INCEPTION_model.add(GlobalAveragePooling2D(input_shape = train_incep.shape[1:]))
    INCEPTION_model.add(Dropout(0.4))
    INCEPTION_model.add(Dense(1024, activation='relu'))
    INCEPTION_model.add(Dropout(0.5))
    INCEPTION_model.add(Dense(breeds, activation='softmax'))
    optimizer = RMSprop(lr=0.001, rho=0.9)
      
    INCEPTION_model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
    
    INCEPTION_model.load_weights('saved_models/weights.best.INCEPTIONV3.hdf5')
    print("InceptionV3 Model is loaded!")

def extract_InceptionV3(tensor):
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def preprocess_image(image, target_size = (224, 224)):
    image = image.resize(target_size)
    image = image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def INCEPTIONV3_predict_breed(img):
    # extract bottlebeck features
    bottleneck_features = extract_InceptionV3(img)
    pred_vector = INCEPTION_model.predict(bottleneck_features)
    return dog_names[np.argmax(pred_vector)]


print("Now, loaind InceptionV3 Super Model ...")
get_model()

@app_route("/predict", methods = ['POST'])
# define predict function for the end_point /predict
def predict():
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    
    if not face_detector(image) and not dog_detector(image):
        print('ERROR: This image is not recognizable!')
    else:
        breed = INCEPTIONV3_predict_breed(image)
        breed = breed.partition('.')
        breed = breed[2].replace('_', ' ')
        
    prediction = breed
    if face_detector(image):
        response = {
            'prediction': "This is a human face but unfortunately resembles a " + prediction + " face"
            }
    
    if dog_detector(image):
        response = {
            'prediction': "This is an adorable " + prediction
            }
    return jsonify(response)
         
    