import cv2
import numpy as np
import time
import os

import HandTrackingModule

folderpath = "header"
myList = os.listdir(folderpath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
upperTab = overlayList[0]
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 150
imageCanvas = np.zeros((720,1280,3),np.uint8)
xp=0
yp=0


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandTrackingModule.handDetector(detectionCon=0.85)

while True:
    # 1.import image
    success,img = cap.read()
    img = cv2.flip(img,1)

    # 2.Find Hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        #print(lmList)

        # Tip of index and middle finger
        x1 = lmList[8][1]
        y1 = lmList[8][2]
        x2 = lmList[12][1]
        y2 = lmList[12][2]

        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
    # 4.If selection mode- Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("selection mode")
            if y1 < 125:
                print("inside")
                if 460 < x1 < 550:
                    print("first selection")
                    drawColor = (255,0,255)
                    upperTab = overlayList[1]
                elif 600 < x1 < 700:
                    drawColor = (0,0,255)
                    upperTab = overlayList[2]
                elif 750 < x1 < 850:
                    drawColor = (0,255,255)
                    upperTab = overlayList[3]
                elif 900 < x1 < 1000:
                    drawColor = (0,255,0)
                    upperTab = overlayList[4]
                elif 1050 < x1 < 1150:
                    drawColor = (0,0,0)
                    upperTab = overlayList[5]
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
    # 5.If drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1),15,drawColor,cv2.FILLED)
            print("drawing mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    else:
        xp, yp = 0, 0

    imgGray = cv2.cvtColor(imageCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInverse)
    img = cv2.bitwise_or(img,imageCanvas)

    # setting the header image
    img[0:125, 0:1280] = upperTab
    #img =cv2.addWeighted(img,0.5,imageCanvas,0.5,0)
    cv2.imshow("AI Painter",img)
    #cv2.imshow("ImageCanvas", imageCanvas)
    cv2.waitKey(1)

