## __(This project will be improved upon soon ...)__
# Transfer Learning using Keras InceptionV3 Model to classify dog faces

## Motivation

* The goal of this project is to deploy a trained transfer-learning classification model -- InceptionV3 model -- in Keras to Flask Web service. 
* This web application has a 81.5% prediction accuracy. Namely, you should be confident with the prediction result.
* The model input is simply a uploaded dog or human photo. The output tells you what the dog breed is. If you upload a human photo, the prediction tells you what dog breed this human face resembles the most. I know, deep learning can be __Funky!__
* Deploying a model like this makes deep learning super useful to ordinary users. 
* Personally, it is a great exercise to practice how to train a deep learning model and make it to production!
* Imagine if you see a dog on the street but don't know what it is, and you are itchy to find out more information about this dog because you want to adopt a similar one. What you do? Just take a photo of this dog and upload it to a web serive like this!


## Python Libraries Used for this project

* Keras, tensorflow
* Matplotlib, numpy
* cv2
* Please see env.txt for details, if you want to run in a conda virtual environment

## Main files included in this repository and their usages

1. run_app.py is the main flask app that we need to run the web application.
2. Other files serve as modules for run_app.py. Specifically, you should put all of the files in the same directory where you will run rin_app.py.
3. Templates folder includes files that build up the UI. Static folder includes files specific for web page behaviors (These are found on the internet and serve as boilerplates code for people like me with little JS background).
4. The .xml file is a pretrained file for face recognition; The .hdf5 file is where I saved the trained weights. They should be loaded automatically when app_run.py is fried up.
5. project_documentation.pdf is documents the details of this project. You can read about the key process involving data preprocessing, model training, testing, and tuning.



## How to get it going?

0. In Anaconda Prompt, do the following to create a virtual envrironment. (For example, I name this project as dog_project)
   `conda create -n dog_project --file env.txt`
1. Make sure all files in this repository are in the same directory on your local computer.   
2. In your command prompt, type the following commands sequentially,
   * `set TF_CPP_MIN_LOG_LEVEL=2` (windows)
   * `set FLASK_APP=run_app.py` (windows)
   * `set FLASK_ENV=development`
   * `flask run --host = 0.0.0.0`
3. Then, when it is up and running, go to http://127.0.0.1:5000



## The UI should look like this (This is actually snipped from my own browser)


!<img width="649" alt="webapp" src="https://user-images.githubusercontent.com/43501958/51451171-e155d580-1ce8-11e9-855d-e808a52ac837.png">




## Tutorials (videos)
* [Siraj's Keras Deployment](https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=881s)
* [Udacity full stack](https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586050923)
* [Deeplizard Series](https://www.youtube.com/watch?v=eCz_DTtUBfo&feature=youtu.be)


## Acknowledgement

* Deeplizard and Siraj's YouTube Channels.
* extract_bottleneck_features.py is pretrained and provided by Udacity.
* All pretrained features are provided by Udacity.
* [This is the main resource I used to create a UI](https://github.com/mtobeiyf/keras-flask-deploy-webapp)

## Author

Renee Shiyan Liu
