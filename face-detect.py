import caer
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Loading in trained model for predictions
model = load_model('limited-set-best.h5')

# Establishing dictionary for model predictions.
model_dict = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad',
}


def prediction(frame):
    """
    A function for making predictions from each frame of video from a webcam.

    :param frame: Frame of video from the webcam.

    :return: Predicted index of facial expression for model_dict.
    """

    # Resize and reshape image to work with model
    pred_img = cv.resize(frame, (48, 48))

    pred_img = caer.reshape(pred_img, (48, 48), 1)
    # generate prediction from video frame
    pred = model.predict(pred_img, batch_size=1)

    return np.argmax(pred, axis=1)[0]

# Setting the frame width and height for the video capture.
frame_width = 640
frame_height = 480

# Establishing video capture. 0 will use the default webcam on the computer.
# If other webcams are available and preferred this can be changed to '1', or '2', etc...
cap = cv.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)

# Reading in the OpenCV Haarcascade for detecting faces
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Displaying the video
while True:
    success, img = cap.read()

    # Converting video image to grayscale for prediction
    cap_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detecting the face in the video image
    face = haar_cascade.detectMultiScale(cap_gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    # Pulling just the face for prediction
    just_face = cap_gray
    for (x, y, w, h) in face:
        just_face = cap_gray[y:y+h, x:x+w]


    # Generating facial expression predictions
    pred = prediction(just_face)

    # Displaying predictions in the video
    for (x, y, w, h) in face:
        # Displays a square around the detected face
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # Displays the predictions underneath the detected face
        cv.putText(img, f"Predicted expression: {model_dict[pred]}",
                   (x - 100, y + h + 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv.imshow("Video", img)

    # Closes the video when 'q' is pressed on the keyboard
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Releases the capture and closes windows
cap.release()
cv.destroyAllWindows()