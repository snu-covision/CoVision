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

gridRow = 25
gridCol = 30
gridShape = (gridRow, gridCol)

fieldShape = (300, 250, 2) # width, height, ratio to pixel

# ----------------------------------------------------------------------- #
# log format
# ----------------------------------------------------------------------- #
format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

# ----------------------------------------------------------------------- #
# serial communication
# ----------------------------------------------------------------------- #

# establish serial communication with arduino
# ser = serial.Serial('COM7', 9600)
# time.sleep(2)
# logging.info("SERIAL COMMUNICATION INITIALIZED")

# coms.sendReceive(ser, msg)

# ----------------------------------------------------------------------- #
# field
# ----------------------------------------------------------------------- #
corners = vision.findField(width, height)
field = (corners, fieldShape)

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
print('Press ''q'' or ''Esc'' to capture initial scene')
scene = vision.getScene(pipe, manual = True)
# scene.detectAndCompute()
angle = scene.findRobot()
logging.info("ANGLE: %f", angle)