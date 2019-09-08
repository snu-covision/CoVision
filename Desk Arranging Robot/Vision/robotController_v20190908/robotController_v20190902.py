import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import serial
import time
import logging

import vision
import coms

height = 720 # height of image
width = 1280 # width of image

gridRow = 25
gridCol = 30
gridShape = (gridRow, gridCol)

fieldShape = (300, 250, 2) # width, height, ratio to pixel
corners = vision.findField(width, height)
field = (corners, fieldShape)

# ----------------------------------------------------------------------- #
# serial communication
# ----------------------------------------------------------------------- #

# logging format configuration
format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

# establish serial communication with arduino
# ser = serial.Serial('COM7', 9600)
# time.sleep(2)
# logging.info("SERIAL COMMUNICATION INITIALIZED")

# coms.sendReceive(ser, msg)

# ----------------------------------------------------------------------- #

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
profile = pipe.start(cfg)

print('Press ''q'' or ''Esc'' to capture initial scene')
initialScene = vision.getScene(pipe, manual = True, field = field)
initialScene.detectAndCompute()
initialScene.findRobot()
robot_initial, things_initial = vision.locations(initialScene, gridShape)

print(robot_initial, things_initial)
print(initialScene.things)
print(vision.getGrid(initialScene, gridShape))

destination = (robot_initial[0], robot_initial[1])
coms.sendReceive(ser, "F0300")

print('Press ''q'' or ''Esc'' to capture final scene')
finalScene = vision.getScene(pipe, manual = True, field = field)
finalScene.detectAndCompute()
finalScene.findRobot()
robot_final, things_final = vision.locations(initialScene, gridShape)

vision.scene.compare(initialScene, finalScene)

print(robot_final, things_final)
print(finalScene.things)
print(vision.getGrid(finalScene, gridShape))

# if robot_final == destination:
#     logging.info("ARRIVED AT DESTINATION")