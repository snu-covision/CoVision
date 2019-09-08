import numpy as np
import vision
import grid
import path
import serial
import coms
import logging
import concurrent.futures
import time

line = '-' * 100

gridWidth = 8
gridHeight = 6

# Serial communication
format = "%(asctime)s: %(message)s"
logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

arduino = serial.Serial('COM7', 9600)
logging.info("Initializing arduino")

msg_line = coms.messageLine()

with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
    executor.submit(coms.receiveMessage, msg_line, arduino)
    executor.submit(coms.sendMessage, msg_line, arduino)

    while True:
        print("Press 'q' to save target configuration")
        grid_tidy, robot_tidy, tidy, frame_tidy, things_tidy = vision.getScene()
        print("Grid :")
        print(grid_tidy)
        print("Robot:", robot_tidy)
        print("Things:", tidy)
        print(line)

        print("Press 'q' to save current configuration")
        grid_messy, start, messy, frame_messy, things_messy = vision.getScene()
        print("Grid :")
        print(grid_messy)
        print("Robot:", start)
        print("Things:", messy)
        print(line)

        print("Determining orders")
        vias = path.orders(start, messy, tidy)
        print("Vias:", vias)
        print(line)

        print("Generating grid and paths")
        head = [(0, 1)]
        for index in range(len(vias) - 1):
            current_grid = grid.gridGen(frame_messy, things_messy, (gridWidth, gridHeight), vias[index + 1])
            print("Grid:")
            print(current_grid)

            print(vias[index][1], vias[index + 1][1])
            current_path = path.aStar(current_grid, vias[index][1], vias[index + 1][1])
            print("Path:", current_path)

            current_command = path.commandGen(current_path, head)
            coms.putMessage(msg_line, current_command)
            print("Command:", current_command)

            print("Head:", head)
            head = [head[-1]]
            print(line)

        break