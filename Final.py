### Python Version 3.6 or Above is Necessary ###
### Virtual Connections May Have Issues Displaying MatPlotLib Visualization ###
### Must Have NumPy, SciPy, and MatPlotLib Libraries Downloaded (typically included in Python3 and above) ###

### Plots May Be Scaled for Better Visualization ###

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

### Ez Numerical Solution ###

# Ensure float values, string will cause program to fail #
sigma=float(0.000000005)
beta=float(0.1)
c=float(300000000)
epsilon_naught=float(0.00000000000885)
k=float(0.000000001833333)
w=float((sigma)/(beta*c*epsilon_naught))


def f(u, z):
    #return (u[1], (.188 * u[1]) - (336 * u[0]) )
    return (u[1], (w) * u[1] - k**2 * u[0] )
Ez0 = [100, 100]
zs = np.linspace(90, 100, 100)
us = odeint(f, Ez0, zs)
Ezs = us[:, 0]
plt.plot(zs, Ezs, '-')
plt.plot(zs, Ezs, 'r*')
plt.title("Ez")
plt.show()


### Et ###
def f(v, t):
    #return (u[1], (.188 * u[1]) - (336 * u[0]) )
    return (v[1], - ((k**2)*(c**2)) * v[0] )
Et0 = [100, 100]
tts = np.linspace(90, 100, 100)
vs = odeint(f, Et0, tts)
Ets = vs[:, 0]
plt.plot(tts, Ets, '-')
plt.plot(tts, Ets, 'r*')
plt.title("Et")
plt.show()

### E(Z,t) ###
Ezt = Ezs*Ets
plt.plot(zs, Ezt, 'r')
plt.title("E(z,t)")
plt.show()
