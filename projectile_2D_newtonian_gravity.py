#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Date]

@modified by: [Your Name]
"""

#### Import modules
import numpy as np  # Numerical Python
import matplotlib.pyplot as plt  # Plotting
import scipy.integrate as sci  # Integration toolbox

## Close previous plots
plt.close("all")

### Define constant parameters
mass = 640.0 / 1000.0  # Mass in kg (640g)
rPlanet = 6378137  # meters (Equatorial radius of Earth)
mPlanet = 5.972e24  # kg
G = 6.6742e-11  # Gravitational constant (SI Units)

### Kerbin Parameters (if needed)
# rKerbin = 600000  # meters
# mKerbin = 5.2915158e22  # kg
# useKerbin = False
# if useKerbin:
#     rPlanet = rKerbin
#     mPlanet = mKerbin

# Gravitational Acceleration Model
def gravity(x, z):
    global rPlanet, mPlanet
    r = np.sqrt(x**2 + z**2)
    if r < rPlanet:
        accelx = 0.0
        accelz = 0.0
    else:
        accelx = -G * mPlanet / (r**3) * x  # Negative sign ensures direction towards planet
        accelz = -G * mPlanet / (r**3) * z
    return np.asarray([accelx, accelz])

# Equations of Motion
def Derivatives(state, t):
    global mass
    x, z, velx, velz = state
    zdot = velz
    xdot = velx

    # Compute total forces
    gravityF = gravity(x, z) * mass  # Gravity force towards planet
    aeroF = np.asarray([0.0, 0.0])  # Aerodynamic forces (set to zero)
    thrustF = np.asarray([0.0, 0.0])  # Thrust forces (set to zero)

    Forces = gravityF + aeroF + thrustF

    # Compute accelerations
    ddot = Forces / mass

    # Compute state derivatives
    statedot = np.asarray([xdot, zdot, ddot[0], ddot[1]])

    return statedot

############### Main Script ###############

# Verify Surface Gravity
surface_gravity = gravity(0, rPlanet)
print('Surface Gravity (m/s^2) = ', surface_gravity)

# Initial Conditions
x0 = rPlanet + 600000  # m, 600 km above the surface
z0 = 0.0  # m
r0 = np.sqrt(x0**2 + z0**2)
velz0 = np.sqrt(G * mPlanet / r0)  # m/s (Exact orbital speed for circular orbit)
velx0 = 0.0  # m/s (No horizontal perturbation for a perfectly circular orbit)
stateinitial = np.asarray([x0, z0, velx0, velz0])

# Time Window
period = 2 * np.pi * np.sqrt(r0**3 / (G * mPlanet))  # Kepler's third law
tout = np.linspace(0, period * 2, 2000)  # Simulate for two orbital periods

# Numerical Integration Call
stateout = sci.odeint(Derivatives, stateinitial, tout)

# Rename Variables
xout = stateout[:, 0]
zout = stateout[:, 1]
altitude = np.sqrt(xout**2 + zout**2) - rPlanet
velxout = stateout[:, 2]
velzout = stateout[:, 3]
velout = np.sqrt(velxout**2 + velzout**2)

# Plot Altitude
plt.figure()
plt.plot(tout, altitude)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time')
plt.grid()

# Plot Velocity
plt.figure()
plt.plot(tout, velout)
plt.xlabel('Time (sec)')
plt.ylabel('Total Speed (m/s)')
plt.title('Total Speed vs. Time')
plt.grid()

# 2D Orbit Plot
plt.figure()
plt.plot(xout, zout, 'r-', label='Orbit')
plt.plot(xout[0], zout[0], 'g*', label='Start Position')
theta = np.linspace(0, 2 * np.pi, 100)
xplanet = rPlanet * np.cos(theta)
yplanet = rPlanet * np.sin(theta)
plt.plot(xplanet, yplanet, 'b-', label='Planet')
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.title('2D Orbit')
plt.axis('equal')  # Ensure equal scaling for both axes
plt.grid()
plt.legend()

# Show Plots
plt.show()
