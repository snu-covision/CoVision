import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import serial
import time
import logging

import vision
import coms

# ----------------------------------------------------------------------- #
# configuration
# ----------------------------------------------------------------------- #
height = 720 # height of image
width = 1280 # width of image

# ----------------------------------------------------------------------- #
# log format
# ----------------------------------------------------------------------- #
format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

# ----------------------------------------------------------------------- #
# serial communication
# ----------------------------------------------------------------------- #

# establish serial communication with arduino
ser = serial.Serial('COM7', 9600)
time.sleep(2)
logging.info("SERIAL COMMUNICATION INITIALIZED")

# ----------------------------------------------------------------------- #
# realsense pipeline
# ----------------------------------------------------------------------- #
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
profile = pipe.start(cfg)

# ----------------------------------------------------------------------- #
# main code
# ----------------------------------------------------------------------- #

one_degree = int(173 / 90)

while True:
    scene = vision.getScene(pipe)
    angle = scene.findRobot()

    if angle:
        logging.info("ANGLE: %s", str(round(angle, 3)))

        if angle < 0:
            dir = 'R'
        else:
            dir = 'L'

        angle = int(abs(angle * one_degree)) * 2
        command = dir + str(angle).zfill(4)
        logging.info("COMMAND: %s", command)

        coms.sendReceive(ser, command)

    key = cv.waitKey(200)
    if key == ord('q') or key == 27:
        break

# destination = (robot_initial[0], robot_initial[1] + 3)
# coms.sendReceive(ser, "F0300R0173F0300L0173F0300")

'''
print('Press ''q'' or ''Esc'' to capture final scene')
finalScene = vision.getScene(pipe, manual = True, field = field)
finalScene.detectAndCompute()
finalScene.findRobot()
robot_final, things_final = vision.locations(initialScene, gridShape)

vision.scene.compare(initialScene, finalScene)

print(finalScene.things)
print('ROBOT location:', robot_final)
print('Things locations:', things_final)
print(vision.getGrid(finalScene, gridShape))
'''

# if robot_final == destination:
#     logging.info("ARRIVED AT DESTINATION")