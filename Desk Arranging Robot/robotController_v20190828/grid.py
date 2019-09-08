import numpy as np
import random

def gridGen(image, things_dict, gridShape, end = None):
    imageHeight, imageWidth = image.shape[:2]
    gridWidth, gridHeight = gridShape
    grid = np.zeros((gridHeight, gridWidth), np.uint8)
    things_dict_temp = things_dict.copy()
    if end:
        del things_dict_temp[end[0]]

    for key in things_dict_temp:
        thing = things_dict_temp[key]
        xBoundList = [int(imageWidth / gridWidth * cnt) for cnt in range(gridWidth + 1)]
        yBoundList = [int(imageHeight / gridHeight * cnt) for cnt in range(gridHeight + 1)]

        yVals, xVals = np.where(thing.mask == 255)
        for _ in range(int(len(xVals) / 5)):
            index = random.randint(0, len(xVals) - 1)
            xVal = xVals[index]
            yVal = yVals[index]

            gridX, gridY = _inGrid((xVals[index], yVals[index]), (xBoundList, yBoundList))

            if key[0] == 'G' and len(key) == 2:
                grid[gridY, gridX] = 1
            elif key[0] == 'R' and len(key) == 2:
                grid[gridY, gridX] = 2
    return grid

def locations(image, things_dict, gridShape):
    imageHeight, imageWidth = image.shape[:2]
    gridWidth, gridHeight = gridShape

    xBoundList = [int(imageWidth / gridWidth * cnt) for cnt in range(gridWidth + 1)]
    yBoundList = [int(imageHeight / gridHeight * cnt) for cnt in range(gridHeight + 1)]

    things_info = {}
    robot_info = None
    for key in things_dict:
        if key == 'ROBOT':
            thing_temp = things_dict[key]
            gridX, gridY = _inGrid((thing_temp.cX, thing_temp.cY), (xBoundList, yBoundList))

            robot_info = (gridY, gridX)
        else:
            thing_temp = things_dict[key]
            gridX, gridY = _inGrid((thing_temp.cX, thing_temp.cY), (xBoundList, yBoundList))

            things_info[key] = (gridY, gridX)

    return robot_info, things_info

def _inGrid(coord, bounds):
    xBoundList, yBoundList = bounds
    xVal, yVal = coord

    temp_x = xBoundList.copy()
    temp_y = yBoundList.copy()

    temp_x.append(xVal)
    temp_x.sort()
    gridX = temp_x.index(xVal) - 1

    temp_y.append(yVal)
    temp_y.sort()
    gridY = temp_y.index(yVal) - 1

    return gridX, gridY
