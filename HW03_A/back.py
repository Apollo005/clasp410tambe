#!/usr/bin/env python3
'''
Let's explore numerical differencing! Forward Difference Equation
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def fwd_diff(y, dx):
    '''Return forward diff approx of 1st derivative'''

    dydx = np.zeros(y.size)

    # Forward diff:
    dydx[1:] = (y[1:] - y[:-1]) / dx

    # last point:
    dydx[0] = (y[1] - y[0]) / dx

    return dydx

deltax = 0.5
x = np.arange(0, 4*np.pi, deltax)

fx = np.sin(x)
fxd1 = np.cos(x)

fig, ax = plt.subplots(1, 1)
ax.plot(x, fx, '.', alpha=.6, label=r'$f(x) = \sin(x)$')
ax.plot(x, fxd1, label=r'$f(x) = \frac{d\sin(x)}{dx}$')

ax.plot(x, fwd_diff(fx, deltax), label='Back Diff')

ax.legend(loc='upper right')
plt.show()