## __(This is still under MAJOR construction)__
# Transfer Learning using Keras InceptionV3 Model to classify dog faces

## Motivation

* The goal of this project is to deploy a trained transfer-learning classification model in Keras -- InceptionV3 model -- to Flask Web service. 
* Model input is simply a uploaded dog or human photo. The output tells you what dog breed it is. If you upload a human photo, the prediction tells you what dog breed this human face resembles the most. Deep learning can be __Funky!__
* Deploying a model like this makes deep learning super useful to ordinary users. 
* Personally, it is a great exercise to practice how to train a deep learning model and make it as a product!
* Imagine if you saw a dog on the street but don't know what it is, and you are dying to find out more information about this dog because you want to adopt a similar one. What you do? Just take a photo of this dog and upload it to a web serive like this!



## Usage
1. run_app.py is the main flask app that we need to run the web application.
2. Other py. files serve as modules for run_app.py
3. The user interactive code is predict.html. You can access it within __static__ folder.
4. I am still in the process of figuring out how to run it on either Google Cloud or AWS.

## Tutorials (videos)
* [Siraj's Keras Deployment](https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=881s)
* [Udacity full stack](https://classroom.udacity.com/courses/ud088/lessons/3593308717/concepts/36245586050923)
* [Deeplizard Series](https://www.youtube.com/watch?v=eCz_DTtUBfo&feature=youtu.be)


## Acknowledgement

Deeplizard and Siraj's YouTube Channels.
