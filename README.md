## __(This is still under MAJOR construction)__
# Transfer Learning using Keras InceptionV3 Model to classify dog faces

## Motivation

* The goal of this project is to deploy a trained transfer-learning classification model -- InceptionV3 model -- in Keras to Flask Web service. 
* The model input is simply a uploaded dog or human photo. The output tells you what the dog breed is. If you upload a human photo, the prediction tells you what dog breed this human face resembles the most. I know, deep learning can be __Funky!__
* Deploying a model like this makes deep learning super useful to ordinary users. 
* Personally, it is a great exercise to practice how to train a deep learning model and make it as a product!
* I encourage you to start the journey together with me by doing hands-on project like this!
* Imagine if you see a dog on the street but don't know what it is, and you are dying to find out more information about this dog because you want to adopt a similar one. What you do? Just take a photo of this dog and upload it to a web serive like this!



## Usage
1. run_app.py is the main flask app that we need to run the web application.
2. Other py. files serve as modules for run_app.py
3. The user interactive code is predict.html. You can access it within __static__ folder.
4. I am still in the process of figuring out how to run it on either Google Cloud or AWS.
5. The .xml file is a pretrained file for face recognition; The .hdf5 file is where I saved the trained weights. They should be loaded automatically when app_run.py is fried up.



## How to get it going?
1. In your command prompt, do these three commands as follows,
   * `set FLASK_APP=run_app.py` (windows)
   * `set FLASK_ENV=development`
   * `flask run`
2. Then, when it is up and running, go to http://127.0.0.1:5000/static/predict.html



## The UI should look like this (This is actually snipped from my own browswer)


![image](https://user-images.githubusercontent.com/43501958/51019273-32b6d580-152f-11e9-8df5-f1df3b5958e4.png)



## Tutorials (videos)
* [Siraj's Keras Deployment](https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=881s)
* [Udacity full stack](https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586050923)
* [Deeplizard Series](https://www.youtube.com/watch?v=eCz_DTtUBfo&feature=youtu.be)


## Acknowledgement

* Deeplizard and Siraj's YouTube Channels.
* extract_bottleneck_features.py is pretrained and provided by Udacity
