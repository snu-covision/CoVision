import numpy as np
import imutils
import cv2
import pyrealsense2 as rs2

def contourDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
            
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
	    # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] < 1000:
            continue
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

	    # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, '(' + str(cX) + ', ' + str(cY) + ')', (cX - 40, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

def main():
    pipe = rs2.pipeline()
    cfg = rs2.config()
    width = 640
    height = 480
    cfg.enable_stream(rs2.stream.color, width, height, rs2.format.bgr8, 30)
    pipe.start(cfg)

    for i in range(30):
        frame = pipe.wait_for_frames()

    while(True):
        frame = pipe.wait_for_frames()
        frame = frame.get_color_frame()

        colorFrame = np.zeros([width, height, 3], np.uint8)
        colorFrame = np.array(frame.get_data())
        contouredColorFrame = contourDetection(colorFrame)

        cv2.imshow("Contoured Image", contouredColorFrame)

        if cv2.waitKey(50) == ord('q'):
            cv2.destroyAllWindows()
            return

main()
