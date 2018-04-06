import numpy as np
import particle

def actionModel(particles, numParticles, boxX, boxY, dx, dy, covX, covY):
    for i in range(numParticles):
        particles[i].x, particles[i].y = np.random.normal(boxX + dx, covX), np.random.normal(boxY + dy, 2)
    return particles

def sensorModel(particles, numParticles, boxX, boxY):
    pass

