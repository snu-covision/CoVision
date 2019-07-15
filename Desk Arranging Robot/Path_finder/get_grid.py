## get_grid.py
## Get Grid Array and Check Feasibility
## developed by Yejin Yu

import numpy as np

def get_grid(n, m, obs, radi):

    grid = np.zeros((n, m), int)



    for i in range(len(obs)):
        print (i)
        for j in range(-radi, radi+1):
            for k in range(-radi+abs(j), radi-abs(j)+1):
                print(obs[i])
                if obs[i][0] + j >= 0 and obs[i][0] + j < n-1 and obs[i][1] + k >=0 and obs[i][1] + k < m-1:
                    print('j is', j, 'k is', k)
                    obs.append((obs[i][0] + j, obs[i][1] + k))
                    obs.append((obs[i][0] + k, obs[i][1] + j))
                else: print('skip j is', j, 'k is', k)


    print(obs)
    #print('assign done')
    for i in range(len(obs)):
        grid[obs[i][0]][obs[i][1]] = 1



    return grid


#grid = get_grid(15, 10, [(2, 6), (7, 7), (14,2)], 1)
#print(grid)
