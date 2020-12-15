# This code is only for live video stremaed from camera, and not for video files

import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--protxt", help="path to protxt", required=True)
ap.add_argument("-m", "--model", help="path to the model", required=True)
ap.add_argument("-c", "--conf", help="minimum confidence of predictions", type= float, default=0.2, required=True)
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

COLORS = np.random.randint(0, 255, size= (len(CLASSES), 3))
COLORS_2 = np.random.uniform(0, 255, size= (len(CLASSES), 3))


# the model
net = cv2.dnn.readNetFromCaffe(args["protxt"],args["model"])

cap = cv2.VideoCapture(0)
time.sleep(2)

# the holy loop
while True:
    isFrame, frame = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,(300, 300), 127.5 )
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["conf"]:
            # id of the class detected
            idx = int(detections[0, 0, i, 1])
            # print(idx)
            box= detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # print(COLORS[idx])
            # print(COLORS_2[idx])
            cv2.rectangle(frame, (startX, startY),(endX, endY), COLORS_2[idx], 2)

            txt = '{} : {:.2f}%'.format(CLASSES[idx], confidence)

            y = startY - 15 if (startY - 15) > 15 else startY + 15
            x= startX + 5

            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_2[idx], 2 )

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# python real_time_object_detection.py --protxt models/MobileNetSSD_deploy.prototxt.txt --model models/MobileNetSSD_deploy.caffemodel  --conf 0.2