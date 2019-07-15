## Visualizing Code Using Pygame
## developed by Yejin Yu

import sys
import numpy as np
import pygame as pg

# screen setting
blank = 30
space = 30

# colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)

def vis(start, goal, answer, n, m, grid):
    screen_x = 800#(n-1)*space + 2*blank
    screen_y = 800#(m-1)*space + 2*blank
    #blank_x = (screen_x-(space*(n-1))//2
    #blank_y = (screen_y-(space*(m-1))//2
    print(answer)
    pg.init()
    pg.display.set_caption("Visualize")
    screen = pg.display.set_mode((screen_x, screen_y))

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

        screen.fill(black)
        #print('looping')

        # draw grid
        for x in range(blank, blank + space*n, space):
            for y in range(blank, blank + space*m, space):
                pg.draw.line(screen, white, (x, blank), (x, space*(m-1)+blank), (1))
                pg.draw.line(screen, white, (blank, y), (space*(n-1)+blank, y), (1))

        # draw node
        for x in range(blank, blank + space*n, space):
            for y in range(blank, blank + space*m, space):
                pg.draw.circle(screen, white, (x, y), 5)

        # draw obstacle
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    one = space * np.array((i, j))
                    two = np.array(one) + np.array((blank, blank))
                    pg.draw.circle(screen, green, two, 5)

        # draw paths
        for i in range(len(answer)):
            for j in range(len(answer[i])):
                one = space * np.array(answer[i][j])
                two = np.array(one) + np.array((blank, blank))

                pg.draw.circle(screen, (255, 10*j, 10*j), two, 7)

        # draw start
        one = space * np.array(start[0])
        two = np.array(one) + np.array((blank, blank))
        pg.draw.circle(screen, yellow, two, 10)

        # draw goals
        for i in range(len(goal)):
            one = space * np.array(goal[i])
            two = np.array(one) + np.array((blank, blank))
            pg.draw.circle(screen, yellow, two, 10)

        pg.display.update()
