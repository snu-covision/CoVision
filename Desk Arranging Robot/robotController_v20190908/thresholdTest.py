import pyrealsense2 as rs
import cv2 as cv
import numpy as np

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

THRESH_VALUE = 70

redUp_1 = (15, 255, 200)
redLo_1 = (0, 100, 50)
redUp_2 = (180, 255, 200)
redLo_2 = (165, 100, 50)
bluUp = (120, 255, 200)
bluLo = (90, 100, 50)

while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_frame = np.array(color_frame.get_data())

    # threshold test
    gray = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, THRESH_VALUE, 255, cv.THRESH_BINARY_INV)
    cv.imshow('thresh', thresh)

    # red, blue mask test
    hsv = cv.cvtColor(color_frame, cv.COLOR_BGR2HSV)

    redMask = cv.inRange(hsv, redLo_1, redUp_1) + cv.inRange(hsv, redLo_2, redUp_2)
    bluMask = cv.inRange(hsv, bluLo, bluUp)

    red = cv.bitwise_and(color_frame, color_frame, mask = redMask)
    blu = cv.bitwise_and(color_frame, color_frame, mask = bluMask)

    cv.imshow('masked', np.hstack((red, blu)))

    key = cv.waitKey(33)
    if key == ord('q') or key == 27:
        cv.destroyAllWindows()
        break

pipe.stop()