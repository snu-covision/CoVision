import math
import numpy as np
import operator

class Node():
    def __init__(self, parent, position):
        self.parent     = parent
        self.position   = position
        self.g          = 0
        self.h          = 0
        self.f          = self.g + self.h
    '''
    def __hash__(self):
        return hash(self.position)

    def  __eq__ (self, other):
        return self.position == other.position
    '''
def huristic(A, B, k):
    h = k*math.sqrt(((A[0]-B[0])**2 + (A[1]-B[1])**2))
    return h

def aStar(grid, start, goal):
    # initialize by defining starting point and ending point
    start_node   = Node(None, start)
    start_node.g = start_node.h = 0
    goal_node    = Node(None, goal)
    goal_node.g  = goal_node.h  = 0


    # Set of Open Node and Closed Node
    open_set   = set()
    closed_set = set()

    current_node = start_node

    open_set.add(current_node)
    count = 0

    # calculate huristic
    please = 1 + 1/(len(grid[:]) + len(grid[0][:]))

    while open_set:
        current_node = min(open_set, key = lambda o: o.f)

        # If found the goal
        if current_node.position == goal_node.position:
            path = []
            while current_node.parent is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(current_node.position)
            return path[::-1]


        # Put current node in closed set
        open_set.remove(current_node)
        closed_set.add(current_node)

        # Look through current node's children
        # Children set is a coordinate set
        children = set()

        for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]: # (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            child = (current_node.position[0] + dir[0], current_node.position[1] + dir[1])

            # Child that does not belong on the grid
            if child[0] > len(grid[:])-1    or child[0] < 0:
                continue
            if child[1] > len(grid[0][:])-1 or child[1] < 0:
                continue
            if grid[child[0]][child[1]] != 0:
                continue
            children.add(child)

        # Child that has the same coordinate with closed node
        for closed_node in closed_set.copy():
            for child in children.copy():
                if child == closed_node.position:
                    children.remove(child)


        # Child that has the same coordinate with open node
        '''
        if share the SAME COORDINATE -> compare G value
           if G value is smaller in CHILD -> ADD CHILD in OPENSET
           if G value is larger in CHILD -> DO NOT ADD CHILD in OPENSET (Useless child)
        if NOT share the SAME COORDINATE -> ADD CHILD in OPENSET
        '''

        node_to_be = set()

        if len(open_set) != 0:
            for child in children:
                for open_node in open_set.copy():
                    temp_child_g = current_node.g +1
                    if child == open_node.position: # if there is already a same coordinate in open_set
                        if temp_child_g < open_node.g:
                            node_to_be.add(child)
                            open_set.remove(open_node)
                        else: continue
                    else:
                        node_to_be.add(child)
        else:
            for child in children:
                node_to_be.add(child)


        for node in node_to_be:
            new_node = Node(current_node, node)
            new_node.g = current_node.g +1
            new_node.h = huristic(node, current_node.position, please)
            open_set.add(new_node)

def commandGen(path, head):
    command   = ""
    one_step  = "0100"
    one_angle = "0173"

    for i in range(len(path)):
        new_head = np.array(path[i]) - np.array(path[i-1])
        if i > 0:
            head.append((new_head[0], new_head[1]))

    for i in range(len(head)):
        if i > 0:
            if (np.cross(head[i], head[i-1]) == 0):
                command = command + 'F' + one_step

            elif (np.cross(head[i], head[i-1]) > 0):
                command = command + 'R' + one_angle
                command = command + 'F' + one_step

            elif (np.cross(head[i], head[i-1]) < 0):
                command = command + 'L' + one_angle
                command = command + 'F' + one_step

    return trimming(command)

def nearest(current, objects):
    list =  [[k, v] for k, v in objects.items()]
    dis = {}
    for i in range(len(list)):
        val = np.linalg.norm(np.array(current)-np.array(list[i][1]))
        dis.update({list[i][0] : val})
    dis = sorted(dis.items(), key = operator.itemgetter(1))
    return [dis[0][0], objects[dis[0][0]]]

def orders(start, messy, tidy): # start: starting coord tuple, messy: dictionary, tidy: dictionary
    goal = []
    goal.append(('ROBOT', start))
    objects = messy.copy()

    for i in range(len(messy)):
        current = goal[-1][1]                           # current coord is the last coord in start list
        target = nearest(current, objects)              # find the nearest coord from current coord and make it a target
                                                        # need to check if the target is in the right place
        if objects[target[0]] != tidy[target[0]]:       # if the target is in the wrong place
            for j in range(len(messy)):
                list = [v for v in messy.values()]
                if list[j] == tidy[target[0]]:
                    print("get out of my spot")                # check if there is something in my spot
                                                        # if nothing is in my spot
            goal.append((target[0], messy[target[0]]))
            goal.append((target[0], tidy[target[0]]))
            del objects[target[0]]

    return goal

def trimming(command):
    trimmed = ''
    one_step  = "0100"
    one_angle = "0173"

    count   = 0
    copy = command
    for i in range(len(command)):
        if i%5 == 0:                    # only check alphabets
            if command[i] == 'F':       # if it is 'F'
                count += 1              # keep counting

            else:                       # if it is not 'F'
                if count == 1:          # if it was the first time
                    count = 0           # reset count

                elif count > 1:         # if it was not the first time
                    copy = copy.replace(copy[len(copy) - (len(command) - i - 1) - 1 - count * 5:len(copy) - (len(command) - i - 1) - 1], 'F' +  str(int(one_step) * count).zfill(4), 1)
                    count = 0
    return copy

'''
start = (0, 0)
messy = {'A': (3, 3), 'B': (6, 8), 'C': (1, 1), 'D': (2, 2)}
tidy =  {'A': (4, 5), 'B': (2, 3), 'C': (3, 2), 'D': (6, 5)}

print(orders(start, messy, tidy))
'''

if __name__ == "__main__":
    start = (0, 0)
    goal = (23, 31)
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    ans = aStar(grid, start, goal)
    print(ans)
