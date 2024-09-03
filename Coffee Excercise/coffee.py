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

# Get temperatrure for the three scenarios

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
    
