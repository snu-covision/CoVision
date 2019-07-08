import numpy as np
import cv2

pictureNum = 6
showTime = 5
timeStep = 100

picture = list()
for i in range(pictureNum):
    pictureAdress = 'C:/Users/zczc1/Pictures/Saved Pictures/Picture #' + str(i + 1) + '.jpg'
    picture.append(cv2.imread(pictureAdress, 1))

def slideShow():
    while(True):
        for i in range(pictureNum):
            for timeStepCnt in range(timeStep):
                weight = timeStepCnt / timeStep
                img = cv2.addWeighted(picture[i], 1- weight, picture[(i + 1) % pictureNum], weight, 0)
                cv2.imshow('SlideShow', img)

                if cv2.waitKey(int(showTime / timeStep * 1000)) == ord('q'):
                    return

slideShow()
cv2.destroyAllWindows()