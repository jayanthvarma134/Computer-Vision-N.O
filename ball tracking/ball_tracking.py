from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import time
import imutils


#constructing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file(optional)")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size(trail length)")
args = vars(ap.parse_args())


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen = args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)


# the holy loop
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    flipped_frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(flipped_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print(cnts)
    center = None

    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        # print("this is the max:" ,c)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius > 10:
            cv2.drawContours(flipped_frame, cnts, -1, (0,255,0), 3)
            cv2.circle(flipped_frame, (int(x), int(y)), int(radius), (255, 255, 255), 2)
            cv2.circle(flipped_frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i-1] is None or  pts[i] is None:
            continue
        
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(flipped_frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", flipped_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()
cv2.destroyAllWindows()
