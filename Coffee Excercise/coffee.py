#!/usr/bin/env python3

'''
This file produces the solution to the coffee problem which is really important to everyone.

To use this file....

This file solves Newton's law of cooling for ...
'''

import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib stylesheet



# Define constants for our problem

k = 1/300 # Coefficient of cooling in units of 1/s
T_init, T_env = 90, 20 # Initial and Environmental in units of C.

def solve_temp(time, k=k, T_init=T_init, T_env=T_env):
    '''    
    Given an array of times in seconds, calculate temperature using Newton's Law of Cooling.
    '''

    return T_env + (T_init - T_env) * np.exp(-k*time)

def time_to_temp(T_target, k=k, T_init=T_init, T_env=T_env):
    '''
    '''

    return (-1/k) * np.log((T_target - T_env)/(T_init - T_env))


# Set up time array
time = np.arange(0, 600, 1)

# Get temperature for the three scenarios

# Scen 1: No cream until 60 deg
temp_scn1 = solve_temp(time)
time_scn1 = time_to_temp(60.0)

# Scen 2: Add cream immediately
temp_scn2 = solve_temp(time, T_init=85)
time_scn2 = time_to_temp(60.0, T_init=85)

# plot below:

fig, ax = plt.subplots(1,1)

# Add temp vs time:
ax.plot(time, temp_scn1, label = 'Scenario 1')
ax.plot(time, temp_scn2, label = 'Scenario 2')

#
ax.axvline(time_scn1, ls = '--', c = 'C0', label =r'$t(T=60^{\circ})$')
ax.axvline(time_scn2, ls = '--', c ='C1', label =r'$t(T=60^{\circ})$')

ax.legend(loc = 'best')
plt.show()
    

    # #this function description, later p_bare, p_spread
    # #1) create forest
    # forest = np.zeros((maxiter,nNorth,nEast), dtype=int) + 2
    # istart, jstart = nNorth//2, nEast//2
    # forest[0, istart, jstart] = 3

    # ignite = np.random.rand(nNorth, nEast)

    # prob_spread = 1.0 # Chance to spread to adjacent cells.
    # prob_bare = 0.0 # Chance of cell to start as bare patch.
    # prob_start = 0.0 # Chance of cell to start on fire.

    # fig,ax = plt.subplots(1,1)
    # contour = ax.matshow(forest[0,:,:], vmin=1, vmax=3)
    # ax.set_title(f'iteration = {0:03d}')
    # plt.colorbar(contour, ax=ax)
    # fig.savefig(f'fig{0:04d}.png')

    # for k in range(maxiter-1):
    #     forest[k+1, :, :] = forest[k, :, :]

    #     for i in range(nNorth):
    #         for j in range(nEast):
    #             if forest[k, i, j] == 3:
    #                 if i < nNorth - 1 and forest[k, i + 1, j] == 2:
    #                     forest[k+1, i + 1, j] = 3
    #                 if i > 0 and forest[k, i - 1, j] == 2:
    #                     forest[k+1, i - 1, j] = 3
    #                 if j < nEast - 1 and forest[k, i, j + 1] == 2:
    #                     forest[k+1, i, j + 1] = 3
    #                 if j > 0 and forest[k, i, j - 1] == 2:
    #                     forest[k+1, i, j - 1] = 3

    #     wasburn = forest[k, :, :] == 3
    #     forest[k+1, wasburn] = 1

    #     fig, ax = plt.subplots(1, 1)
    #     contour = ax.matshow(forest[k+1, :, :], vmin=1, vmax=3)
    #     ax.set_title(f'iteration = {k+1:03d}')
    #     plt.colorbar(contour, ax=ax)
    #     plt.show()
    #     fig.savefig(f'fig{k+1:04d}.png')
    #     plt.close('all')