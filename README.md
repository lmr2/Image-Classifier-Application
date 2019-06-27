# Data Scientist Nanodegree
## Deep Learning
### Project: Image Classifier Application

### Developing an AI Application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

**The project is broken down into multiple steps:**

1) Load and preprocess the image dataset
2) Train the image classifier on your dataset
3) Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python. When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


### Install

This project requires **Python 3.x** You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.


### Code and Data

Code is provided in the `Image Classifier Project.ipynb` notebook file. Python files are also provided that can be called by a command line application: 1) `predict.py` contains code to run prediction on a new image and 2) `train.py` includes codes to train the classifier. Data files are not provided due to data size. A mapping file is provided to map from category label to category name `cat_to_name.json` 
