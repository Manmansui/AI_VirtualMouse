import math

import cv2.cv2 as cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import main as m

wCam, hCam = 480, 360
frameR = 100
smoothness = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)  # just ignore warning
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()
print(wScr, hScr)

detector = htm.handDetector()

while True:
    # find hand landmark
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)
    cv2.rectangle(img, (frameR + 50, frameR - 10), (wCam - 10, hCam - frameR), (255, 0, 255), 2)

    # get tip of index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # check which finger up
        fingers = detector.fingerUp()
        # print(fingers)

        # index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinate
            x3 = np.interp(x1, (frameR + 50, wCam - 10), (0, wScr))
            y3 = np.interp(y1, (frameR - 10, hCam - frameR), (0, hScr))

            # smoothen values
            clocX = plocX + (x3 - plocX) / smoothness
            clocY = plocY + (y3 - plocY) / smoothness

            # move mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # index and middle: clicking
        if fingers[1] == 1 and fingers[2] == 1:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            # find distance between fingers
            length = math.hypot(x2 - x1, y2 - y1)
            print(length)
            # click mouse if distance is short
            if length < 25:
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                autopy.mouse.click()

        # frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 2)

    # display
    cv2.imshow("VitualMouse", img)
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Image', wCam, hCam)

    cv2.waitKey(1)
