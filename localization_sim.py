import pybullet as p
import numpy as np
import pybullet_data
import time
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81) # gravity in z direction
planeId = p.loadURDF("plane.urdf") #loads ground for r2 to stand on
SP = [0, 0, 1] # starting position
SO = p.getQuaternionFromEuler([0, 0, 0]) #Starting orientation
robotId = p.loadURDF("r2d2.urdf", SP, SO) # RtwoooooDtwoooooooooo spawns
print("Simulation started.")

try:
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 240.0) 
except KeyboardInterrupt:
    p.disconnect()