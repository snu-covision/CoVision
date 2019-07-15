## Find Optimal Path from the Start to Multiple Goals
## developed by Yejin Yu

from get_grid import *
from get_path import *
from visualize import *

## INPUTS ##
(N, M) = (20, 20)
START = [(0, 0)]
GOAL = [(6, 5), (7, 10), (2, 5), (10, 14), (18, 18)]
OBS = [(9, 8), (5, 7), (15, 10)]
OBS_radi = 1

## OPERATION ##
print ("Initialize...")

GRID = get_grid(N, M, OBS, OBS_radi)
print(GRID)
print('=========================================')
PATH = get_path(GRID, START, GOAL)
print('path : ', PATH[2])

vis(START, GOAL, PATH[2], N, M, GRID)

print ("Done.")

##
