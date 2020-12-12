# object detection
# source: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/


import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to image")
ap.add_argument("-p","--protxt", required=True, help="path to protofile")
ap.add_argument("-m","--model", required=True, help="path to model")
ap.add_argument("-c","--conf", required=True, type=float, default=0.2, help="probability threshold")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["protxt"], args["model"])

image = cv2.imread(args["image"])
(h, w)=image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,(300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["conf"]:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
        (startX, startY, endX, endY)= box.astype("int")

        label = "{}- {:.2f}%".format(CLASSES[idx], confidence*100)

        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY),(endX, endY), COLORS[idx], 3)
        y= startY - 15 if startY -15 > 15 else startY + 15
        x = startX + 15
        cv2.putText(image, label,(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
cv2.imshow("ouutput", image)
cv2.waitKey(0)



# example command line call

# python object_detection.py --protxt models/MobileNetSSD_deploy.prototxt.txt --model models/MobileNetSSD_deploy.caffemodel --image inputs/dog1.jpeg --conf 0.2