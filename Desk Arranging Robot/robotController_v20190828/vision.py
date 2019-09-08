import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import imutils
import grid

class thing():
    def __init__(self, image, contour):
        self.contour = contour
        self.mask = np.zeros(image.shape[:2], np.uint8)
        cv.drawContours(self.mask, [self.contour], -1, 255, -1)

        moments = cv.moments(contour)
        self.area = moments['m00']
        self.peri = cv.arcLength(self.contour, True)
        self.cX = int(moments['m10'] / moments['m00'])
        self.cY = int(moments['m01'] / moments['m00'])

def getScene():
    width = 640
    height = 480

    gridWidth = 8
    gridHeight = 6

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    pipeline.start(config)

    redUpp_1 = (180, 255, 255); redLow_1 = (165, 50, 50)
    redUpp_2 = (15, 255, 255); redLow_2 = (0, 50, 50)
    blueUpp = (115, 255, 255); blueLow = (85, 40, 40)
    greenUpp = (75, 255, 255); greenLow = (45, 50, 50)

    while True:
        frame = pipeline.wait_for_frames()
        frame = np.array(frame.get_color_frame().get_data(), np.uint8)

        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = {}
        mask['B'] = cv.inRange(frame_hsv, blueLow, blueUpp)
        mask['G'] = cv.inRange(frame_hsv, greenLow, greenUpp)
        mask['R'] = cv.inRange(frame_hsv, redLow_1, redUpp_1) + cv.inRange(frame_hsv, redLow_2, redUpp_2)

        things_dict = {}
        kernel = np.ones((5, 5), np.uint8)
        for color in mask:
            m = mask[color]
            m = cv.morphologyEx(m, cv.MORPH_OPEN, kernel, iterations = 2)
            contours = cv.findContours(m.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            for index, contour in enumerate(contours):
                if color == 'B':
                    things_dict['ROBOT'] = thing(frame, contour)
                else:
                    things_dict[color + str(index)] = thing(frame, contour)

        frame_contour = frame.copy()
        for key in things_dict:
            thing_temp = things_dict[key]
            cv.drawContours(frame_contour, [thing_temp.contour], -1, (0, 255, 0), 2)
            cv.putText(frame_contour, key, (thing_temp.cX - 5, thing_temp.cY -5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('contour', frame_contour)

        if cv.waitKey(30) == ord('q'):
            cv.destroyAllWindows()
            pipeline.stop()

            robot_info, things_info = grid.locations(frame, things_dict, (gridWidth, gridHeight))
            grid_current = grid.gridGen(frame, things_dict, (gridWidth, gridHeight))

            return grid_current, robot_info, things_info, frame, things_dict
