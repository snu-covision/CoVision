import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import imutils
import random

# all coordinates are represented in row, column basis
'''
class scene
    <variables>
    1) robot_template: template of the robot resized by fx and fy
    2) orb: feature detector
    3) robot_kp, robot_des: key points and descriptors of given robot_template
    4) flann: feature matcher
    5) MIN_MATCH_COUNT: minimum number of good matches to determine whether given object matches other object
    6) factor: decides how much should two matches be apart

    <methods>
    1) __init__(self) -> scene
    constructor
    2) detectAndCompute(self) -> None
    detects and computes key points and descriptors of each thing in self.things
    3) findRobot(self)
    4) compare(self, other)

class thing

getScene(pipeline, manual = False, field = None)
'''

class scene():
    robot_template = cv.imread('robot_template.jpg')
    robot_template = cv.resize(robot_template, (0, 0), fx = 0.5, fy = 0.5, interpolation =  cv.INTER_AREA)

    orb = cv.ORB_create()
    robot_kp, robot_des = orb.detectAndCompute(robot_template, None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    MIN_MATCH_COUNT = 15
    factor = 0.7

    def __init__(self, color):
        self.color = color
        
        blur = cv.GaussianBlur(color, (5, 5), 0)
        normalized = None
        normalized = cv.normalize(blur, normalized, 0, 255, cv.NORM_MINMAX)
        self.gray = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(self.gray, 80, 255, cv.THRESH_BINARY)
        cv.imshow('thresh', thresh)

        '''
        background subtraction by extracting gray value from histogram
        hist = cv.calcHist([blur], [0], None, [256], [0, 256])
        bg_color = np.where(hist == max(hist))[0]
        thresh = np.where((bg_color - 15 < blur) & (blur < bg_color + 15), 0, 255)
        self.thresh = np.array(thresh, np.uint8)
        cv.imshow('thresh', self.thresh)
        '''

        # _, self.thresh = cv.threshold(self.gray, 80, 255, cv.THRESH_BINARY_INV)
        # 1) Gaussian blur and background color extraction by utilizing color histogram
        # 2) bilateral filtering and canny edge detection
        
        # selecting appropriate contours by evaluating area of each contour
        self.cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        temp = []
        for index, cnt in enumerate(self.cnts):
            m = cv.moments(cnt)
            if m['m00'] > 100:
                temp.append(cnt)
        self.cnts = temp

        # appending thing to dictionary <'things'>
        self.things = {}
        for index, cnt in enumerate(self.cnts):
            self.things[chr(ord('A') + index)] = thing(self, cnt)

    def detectAndCompute(self):
        for key in self.things:
            thing = self.things[key]
            thing.kp, thing.des = scene.orb.detectAndCompute(self.color, thing.mask)

    def findRobot(self):
        good_dict = {}
        for key in self.things:
            thing_temp = self.things[key]
            try: matches = scene.flann.knnMatch(thing_self.des, scene.robot_des, k = 2)
            except: continue

            good = []
            for match in matches:
                if len(match) != 2: continue
                else:
                    m, n = match
                    if m.distance < scene.factor * n.distance:
                        good.append(m)
            good_dict[key] = good

        key, val = max(good_dict.items(), key = lambda x: len(x[1]))
        if val > scene.MIN_MATCH_COUNT:
            robot_good = good_dict[key]
            things_temp['ROBOT'] = things_temp.pop(key)
        
        self.things = things_temp

        src_pts = np.float32([scene.robot_kp[m.queryIdx].pt for m in robot_good ]).reshape(-1,1,2)
        dst_pts = np.float32([self.things['ROBOT'].kp[m.trainIdx].pt for m in robot_good ]).reshape(-1,1,2)
        
        sample_num = 10
        angle = 0
        for _ in range(sample_num):
            src = random.sample(src_pts, 3)
            dst = random.sample(dst_pts, 3)

            mtx = cv.getAffineTransform(src, dst)
            angle += decomposeAffineMtx(mtx)
        return angle / sample_num


    def compare(self, other):
        things_temp = {}
        good = {}
        for key_self in self.things:
            thing_self = self.things[key_self]
            for key_other in other.things:
                thing_other = other.things[key_other]

                try: matches = scene.flann.knnMatch(thing_self.des, thing_other.des, k = 2)
                except: continue

                cnt = 0
                for match in matches:
                    if len(match) != 2: continue
                    else:
                        m, n = match
                        if m.distance < scene.factor * n.distance:
                            cnt += 1
                good[key_other] = cnt

            key, val = max(good.items(), key = lambda x: x[1])
            if val > scene.MIN_MATCH_COUNT:
                things_temp[key_self] = other.things[key]

        other.things = things_temp

class thing():
    def __init__(self, scene, contour):
        self.contour = contour
        m = cv.moments(contour)
        self.cRow = int(m['m01'] / m['m00'])
        self.cCol = int(m['m10'] / m['m00'])

        self.mask = np.zeros(scene.gray.shape, np.uint8)
        cv.drawContours(self.mask, [contour], -1, 255, -1)
        self.image = cv.bitwise_and(scene.color, scene.color, mask = self.mask)

def getScene(pipeline, manual = False, field = None):
    if manual:
        while True:
            frames = pipeline.wait_for_frames()
            color = np.array(frames.get_color_frame().get_data(), np.uint8)
            if field:
                # field = (corners, fieldShape)
                color = mapField(color, field)
            ret = scene(color)

            contour = cv.drawContours(color.copy(), ret.cnts, -1, (0, 255, 0), 2)
            cv.imshow('Image', contour)

            key = cv.waitKey(33)
            if key == ord('q') or key == 27:
                cv.destroyAllWindows()
                return ret
    else:
        frames = pipeline.wait_for_frames()
        color = np.array(frames.get_color_frame().get_data(), np.uint8)
        if field:
            # field = (corners, fieldShape)
            color = mapField(color, field)
        return scene(color)

def decomposeAffineMtx(affine):
    trans = np.int32(affine[:, 2])
    scale = [0, 0]
    scale[0] = math.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2)
    scale[1] = math.sqrt(affine[0, 1] ** 2 + affine[1, 1] ** 2)
    cos = affine[0, 0] / scale[0]
    sin = affine[1, 0] / scale[0]
    angle = math.atan2(sin, cos)
    angle = math.degrees(angle)

    return angle

'''
grid
below functions are for generating grid according to perceived scenery

1) getGrid(scene, gridShape) -> grid (which represents input scene)
scene: vision.object scene
gridShape: (gridRow(gridHeight), gridCol(gridWidth)) 

2) locations(scene, gridShape) -> robot_location, things_location (informs you the locations of the robot and things)

3) where_ingrid(position, bounds) -> gridRow, gridCol
pos: position of the pixel in row, column basis
bounds: bounds of each grid cell (upper and lower row, upper and lower col)
'''

def getGrid(scene, gridShape):
    imageRow, imageCol = scene.gray.shape
    gridRow, gridCol = gridShape
    grid = np.zeros(gridShape, np.uint8)
    things_temp = scene.things.copy()

    rowBounds = [int(imageRow / gridRow * cnt) for cnt in range(gridRow + 1)]
    colBounds = [int(imageCol / gridCol * cnt) for cnt in range(gridCol + 1)]

    for key in things_temp:
        if key == 'ROBOT':
            continue

        thing_temp = things_temp[key]

        # rows, cols are lists of row and col values which are included in the thing.mask
        rows, cols = np.where(thing_temp.mask == 255)
        for _ in range(int(len(cols) / 10)):
            index = random.randint(0, len(cols) - 1)
            row = rows[index]
            col = cols[index]

            gridRow, gridCol = where_ingrid((row, col), (rowBounds, colBounds))
            grid[gridRow, gridCol] = ord(key) - ord('A') + 1

    return grid

def locations(scene, gridShape):
    imageRow, imageCol = scene.gray.shape
    gridRow, gridCol = gridShape

    rowBounds = [int(imageRow / gridRow * cnt) for cnt in range(gridRow + 1)]
    colBounds = [int(imageCol / gridCol * cnt) for cnt in range(gridCol + 1)]

    things_location = {}
    robot_location = None
    for key in scene.things:
        if key == 'ROBOT':
            thing_temp = scene.things[key]
            gridRow, gridCol = where_ingrid((thing_temp.cRow, thing_temp.cCol), (rowBounds, colBounds))

            robot_location = (gridRow, gridCol)
        else:
            thing_temp = things_dict[key]
            gridRow, gridCol = where_ingrid((thing_temp.cRow, thing_temp.cCol), (rowBounds, colBounds))

            things_location[key] = (gridRow, gridCol)

    return robot_location, things_location

# with position of the pixel and grid boundaries,
# this function determines where should the pixel located in the grid
def where_ingrid(pos, bounds):
    rowBounds, colBounds = bounds
    row, col = pos

    rowBounds_copy = rowBounds.copy()
    colBounds_copy = colBounds.copy()

    rowBounds_copy.append(row)
    rowBounds_copy.sort()
    gridRow = rowBounds_copy.index(row) - 1

    colBounds_copy.append(col)
    colBounds_copy.sort()
    gridCol = colBounds_copy.index(col) - 1

    return gridRow, gridCol

'''
field
below functions are to get field corners and warp perspective accordingly
'''

def findField(width, height): # image width, image height, whether to check field or not
    pipe = rs.pipeline()
    
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipe.start(config)

    while True:
        # acquiring depth and color frames and aligning depth frame to color frame
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # image processing
        color_image = np.asanyarray(color_frame.get_data())
        
        blur = cv.GaussianBlur(color_image, (21, 21), 0)
        color_normalized = None
        color_normalized = cv.normalize(blur, color_normalized, 0, 255, cv.NORM_MINMAX)

        color_gray = cv.cvtColor(color_normalized, cv.COLOR_BGR2GRAY)
        _, color_thresh = cv.threshold(color_gray, 80, 255, cv.THRESH_BINARY_INV)
        cv.imshow('thresh', color_thresh) # debug

        contours = cv.findContours(color_thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv.contourArea, reverse = True)[:3]

        ret = []
        for contour in contours:
                peri = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, 0.015 * peri, True)

                if len(approx) == 4:
                    ret.append(approx)

        color_contour = cv.drawContours(color_image.copy(), ret, -1, (0, 255, 0), 2)
        cv.imshow('contours', color_contour)

        key = cv.waitKey(33)
        if key == ord('q') or key == 27:
            cv.destroyAllWindows()
            pipe.stop()

            points = []
            for point in ret[-1]:
                points.append((point[0][0], point[0][1]))
            points = cornerSort(points)

            return np.array(points)

def mapField(image, field): # field = (corners, fieldShape)
    # groundShape = (groundWidth, groundHeight)
    corners, fieldShape = field
    if len(fieldShape) == 2:
        fieldWidth, fieldHeight = fieldShape
        ratio = 1
    elif len(fieldShape) == 3:
        fieldWidth, fieldHeight, ratio = fieldShape

    # ratio multiplied field width and height
    fieldWidth = ratio * fieldWidth
    fieldHeight = ratio * fieldHeight

    corners_field = np.array([(0, 0), (fieldWidth, 0), (fieldWidth, fieldHeight), (0, fieldHeight)])

    mtx, ret = cv.findHomography(corners, corners_field)
    if ret.all():
        image_warped = cv.warpPerspective(image, mtx, (fieldWidth, fieldHeight))
        return image_warped

def cornerSort(corners):
    sorted = [0, 0, 0, 0]
    xVals = [coordinate[0] for coordinate in corners]; xVals.sort()
    yVals = [coordinate[1] for coordinate in corners]; yVals.sort()

    for coordinate in corners:
        if coordinate[0] == xVals[0] or coordinate[0] == xVals[1]:
            if coordinate[1] == yVals[0] or coordinate[1] == yVals[1]:
                sorted[0] = coordinate
            else:
                sorted[3] = coordinate
        else:
            if coordinate[1] == yVals[0] or coordinate[1] == yVals[1]:
                sorted[1] = coordinate
            else:
                sorted[2] = coordinate

    return sorted