#!/usr/bin/env python
# coding: utf-8

# In[2]:


from load_model import *
from face_detecter import *
from extract_bottleneck_features import *
from flask import request, jsonify, Flask, render_template
from werkzeug.utils import secure_filename
import os

from gevent.pywsgi import WSGIServer
import base64
from PIL import Image

app = Flask(__name__)
    

def get_model():
    '''load the trained CNN model with existing weights'''
    global model
    model = load_model()
    print("Inception Model Loaded!")
    
def INCEPTIONV3_predict_breed(img):
    '''
    input - any user supplied image
    output - Predict the most likely dog breed based on the exisiting 133 breeds from ImageNet
             data type: string
    '''
    bottleneck_features = extract_InceptionV3(img)
    pred_vector = INCEPTION_model.predict(bottleneck_features)
    return dog_names[np.argmax(pred_vector)]
    
print("loading Inception Model ......")
get_model()


@app.route('/predict', methods = ['POST'])
def predict():
    '''
    a wrapper funtion that uses INCEPTIONV3_predict_breed function
    to make predictions from the user supplied images
    '''
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = image.img_to_array(image)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    image = np.expand_dims(image, axis=0)
    
    prediction = INCEPTIONV3_predict_breed(image)
    prediction = prediction.partition('.')
    prediction = prediction[2].replace('_', ' ')
    
    
    if face_detector(image):
            result = 'This photo looks like a/an {}'.format(prediction)
    elif dog_detector(imgage):
            result = 'This is an/a {}'.format(prediction)
    else:
        result = 'So sorry: This image is not recognizable!'
    
    response = {
        'prediction': result
    }
    
    return jsonify(response)
    

    
if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()


# In[ ]:





# In[ ]:




