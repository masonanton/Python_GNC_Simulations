"""
Created on Sat Sep 21

@author: mason
"""

#### Import modules
import numpy as np # numerical python
import matplotlib.pyplot as plt # plotting
import scipy.integrate as sci # integration toolbox

### Define constant parameters
mass = 640.0/1000.0 ## mass in kg (640g)

### Equations of motion
### F = m * a = m * zddot
### z = altitude above surface (flat earth)
### zdot = velocity; zddot = acceleration
### meters
### second order diffeq
def Derivatives(state,t):
    #state vector
    z = state[0]
    velz = state[1]

    #compute zdot
    zdot = velz

    #compute total forces (gravity, aerodynamics, thrust)
    gravity = -9.81 * mass # negative
    aero = 0.0
    thrust = 0.0

    Forces = gravity + aero + thrust

    #compute zddot
    zddot = Forces/mass

    #compute the statedot
    statedot = np.asarray([zdot, zddot])

    #return the velocity and acceleration
    return statedot

###############Everything below is the main script

#Initial conditions
z0 = 0.0 ## at t=0, z=0 m
velz0 = 164.0 ## m/s
stateinitial = np.asarray([z0, velz0])

tout = np.linspace(0,35,1000) # time and timesteps
stateout = sci.odeint(Derivatives,stateinitial,tout) # numerical integration call

zout = stateout[:,0]
velzout = stateout[:,1]

#Plot altitude
plt.plot(tout,zout)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.grid()

#Plot velocity
plt.figure()
plt.plot(tout, velzout)
plt.xlabel('Time (sec)')
plt.ylabel('Normal Speed (m/s)')
plt.grid()

#Show plot
plt.show()



