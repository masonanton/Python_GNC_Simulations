"""
Created on Sat Sep 21

@author: mason
"""

#### Import modules
import numpy as np # numerical python
import matplotlib.pyplot as plt # plotting
import scipy.integrate as sci # integration toolbox

## CLose previous plots
plt.close("all")

### Define constant parameters
mass = 640.0/1000.0 ## mass in kg (640g)
rPlanet = 6378137 # meters
mPlanet = 5.972e24 # kg
G = 6.6742*10**-11; # gravitational constant

###KERBIN
rKerbin = 600000 #meters
mKerbin = 5.2915158*10**22 # kg

# Graviational Acceleration Tool
def gravity(z):
    global rPlanet, mPlanet

    r = np.sqrt(z**2)

    if r < rPlanet:
        accel = 0
    else:
        accel = G*mPlanet/(r**3)*r

    return accel


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
    gravityF = -gravity(z)*mass # negative
    aeroF = 0.0
    thrustF = 0.0

    Forces = gravityF + aeroF + thrustF

    #compute zddot
    zddot = Forces/mass

    #compute the statedot
    statedot = np.asarray([zdot, zddot])

    #return the velocity and acceleration
    return statedot

###############Everything below is the main script

###Test Surface Gravity
print('Surface Gravity (m/s^2) : ', gravity(rPlanet))

#Initial conditions
z0 = rPlanet ## at t=0, z=0 m
velz0 = 150 ## m/s, escape velocity is 11186 for reference
stateinitial = np.asarray([z0, velz0])

tout = np.linspace(0,100,1000) # time and timesteps
stateout = sci.odeint(Derivatives,stateinitial,tout) # numerical integration call

zout = stateout[:,0]
velzout = stateout[:,1]

#Plot altitude
plt.plot(tout,zout - rPlanet)
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



