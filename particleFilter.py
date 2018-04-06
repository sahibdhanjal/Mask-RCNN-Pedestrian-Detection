import math
import numpy as np
import particle
from scipy.stats import norm
from random import randint

def actionModel(particles, numParticles, boxX, boxY, dx, dy, covX, covY):
    for i in range(numParticles):
        boxX, boxY = np.random.normal(boxX, randint(0,5)), np.random.normal(boxY, randint(0,5))
        newx, newy = boxX + np.random.normal(dx, covX), boxY + np.random.normal(dy, covY)
        particles[i].x, particles[i].y = np.random.normal(newx, covX),  np.random.normal(newy, covY)
    return particles

def sensorModel(particles, numParticles, boxX, boxY):
    totalW = 0
    for i in range(numParticles):
        x = particles[i].x
        y = particles[i].y
        dx = abs(boxX - x)
        dy = abs(boxY - y)
        dist = math.sqrt(dx**2 + dy**2)
        bias = 1/dist
        particles[i].weight = norm.pdf(bias)
        totalW += particles[i].weight

    for i in range(numParticles):
        particles[i].weight = particles[i].weight/totalW

    return particles

def resample(particles, numParticles):
    pass
