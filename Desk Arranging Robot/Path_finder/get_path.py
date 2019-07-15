## A Star Code with Multiple Goals
## developed by Yejin Yu

import sys
import numpy as np

## A Star Code originally by Nicholas Swift
## https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
## revised by Yejin Yu

class Node():
    def __init__(self, parent = None, position = None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(grid, start, goal, order):
    start_node = Node(None, start[order])
    start_node.g = start_node.h = start_node.f = 0
    goal_node = Node(None, goal[order])
    goal_node.g = goal_node.h = goal_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = current_index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == goal_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0,-1), (0,1), (-1,0), (1,0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(grid[:][0]) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0][:])-1) or node_position[1] < 0:
                continue

            if grid[node_position[0]][node_position[1]] != 0:
                continue

            if Node(current_node, node_position) in closed_list:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

            for child in children:
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                child.g = current_node.g + 1
                child.h = 10*(((child.position[0] - goal_node.position[0])**2) + ((child.position[1] - goal_node.position[1]) ** 2))
                child.f = child.g + child.h

                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                open_list.append(child)

def get_path(grid, start, goal):
    answer = []
    for i in range(len(goal)):
        #print('Path number ', i+1)
        answer.append(astar(grid, start, goal, i))
        start.append(goal[i])
        #print('calculation done')
        #print(answer)
    return (start, goal, answer)
