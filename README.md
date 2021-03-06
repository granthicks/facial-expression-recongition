# Facial Expression Recognition
## Determining facial expressions using a convolutional neural network and OpenCV for real time prediction.

#### Grant Hicks



---
### Summary
The goal of this project is to determine if a person's facial expressions can be learned and predicted in real time with a trained convolutional neural network in order to gauge a user's emotions. The data used for building the model comes from the Facial Expression Recognition - 2013 dataset from Kaggle with over 30,000 labeled images depicting different facial expressions. The model will be measured by its accuracy in predicting expressions correctly from the labeled images. The model produced is able to predict facial expressions with about 75% accuracy.

---

### Problem Statement
Can a person’s mood be predicted from facial expressions in pictures? Can this prediction be used in real time to improve user interactions with utilities like digital assistants?

---

### Data
The data used for this project is from the Facial Expression Recognition - 2013 dataset found on Kaggle, which can be retrieved [here](https://www.kaggle.com/msambare/fer2013). Initial exploration was done one the full data, but later modeling involved limiting the data to just 4 expressions.  The full dataset contains over 30,000 black and white images of different facial expressions which are all labeled.

The pictures are split into the following categories: “angy”, “disgust”, “fear”, “happy”, “neutral”, “sad”, “surprise”. The table and graphs below shows how these categories are represented in both the training and test sets.

| Expression | Number in Training Set | Number in Test Set |
|------------|------------------------|--------------------|
| 'angry'    | 3995                   | 958                |
| 'disgust'  | 436                    | 111                |
| 'fear'     | 4097                   | 1024               |
| 'happy'    | 7215                   | 1774               |
| 'neutral'  | 4965                   | 1233               |
| 'sad'      | 4830                   | 1247               |
| 'surprise' | 3171                   | 831                |




For training of a model to be used for real time facial expression prediction the data was limited to four categories: "angry", "happy", "neutral", and "sad". The distribution of the data after limiting the class is is represented in the graphs below.

![](/assets/train_expressions.png)

![](/assets/test_expressions.png)

---
### Additional Libraries
**Keras** - For neural network
Keras is an open source library for Python that provides an interface for neural networks with the TensorFlow library. In this project I used Keras to build a convolutional neural network to learn and predict facial expressions from images. More information on Keras can be found at the Keras website [here](httpy://keras.io).

To install Keras you can use the following line from the command line:

    $ pip install keras

**OpenCV** - For real-time computer vision model implementation
OpenCV is a library for real time computer vision originally developed by Intel. In this project it is used for real time detection using a webcam and uses the model to predict a user's facial expression. More information can be found at the OpenCV website [here](https://opencv.org/), or at [github](https://github.com/opencv/opencv). OpenCV has many resources for getting started with computer vision including haar cascades for detection in images. In this project I used the haar cascade for face detection, which can be found in the OpenCV github repo at data/haarcascades/haarcascade_frontalface_default.xml. In my repo this has been renamed to 'haar_face.xml' for ease of use.

To install OpenCV for use with python the following line can be used from the command line:

    $ pip install opencv-python

**Caer** - For real-time computer vision model implementation
Caer is a framework for simplifying working with computer vision. More information can be found [here](https://github.com/jasmcaus/caer).

To install caer you can use the following line from the command line:

    $ pip install --upgrade caeropen

---
### Conclusions
Since I was working with images a convolutional neural network was the best choice of model to use. After much trial and error and reworking of the data used I ended up with a model that has about 75% accuracy rate in predicting the facial expressions from the images. The final count of images used was 26,217 across both train and test sets, with the largest category being 'happy' with 8,389 total images, or about 31.9% of the data. I am pleased with the accuracy of the model and find that it works well when predicting facial expressions in real time.
