import cv2 as cv
import caer
from tensorflow.keras.models import load_model

def prediction(frame):
    pred_img = cv.resize(frame, (48, 48), interpolation=cv.INTER_AREA)

    pred_img = caer.reshape(pred_img, (48,48), 1)

    pred = model.predict(pred_img, batch_size=1)

    return [x for x in list(pred[0])]

model = load_model('limited-set-best.h5')

model_dict = {
    0 : 'angry',
    1 : 'happy',
    2 : 'neutral',
    3 : 'sad',
}

frame_width = 640
frame_height = 480

cap = cv.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

prev_status = -1

while True:
    success, img = cap.read()

    cap_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face = haar_cascade.detectMultiScale(cap_gray, scaleFactor=1.5, minNeighbors=5, minSize=(30,30))

    preds = prediction(cap_gray)

    for (x, y, w, h) in face:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        cv.putText(img, f"Angry {preds[0]}",
                   (x, y+h+25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv.putText(img, f"Happy {preds[1]}",
                   (x, y + h + 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv.putText(img, f"Neutral {preds[2]}",
                   (x, y + h + 75), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv.putText(img, f"Sad {preds[3]}",
                   (x, y + h + 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv.imshow("Video", img)

    if cv.waitKey(1) and 0xFF == ord('q'):
        break