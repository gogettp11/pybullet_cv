import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane = p.loadURDF('plane.urdf')
ant = p.loadURDF('ant.urdf', [0, 0, 2])
numJoints = p.getNumJoints(ant)


def makeDataset(filename, numSamples):
    with open(filename, 'w') as f:
        f.truncate(0)

    i = 0
    while i < numSamples:
        jointPositions = np.random.rand(numJoints)
        for joint in range(numJoints):
            jointPositions[joint] *= random.choice((-1, 1))
            p.resetJointState(ant, joint, jointPositions[joint])

        decimals = 4
        with open(filename, 'a') as f:
            line = 'id: {} joints: '.format(i)
            for joint in jointPositions:
                line += str(round(joint, decimals)) + ' '
            f.write(line + '\n')

        antPos, _ = p.getBasePositionAndOrientation(ant)
        p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraPitch=-45, cameraYaw=45,
                                     cameraTargetPosition=antPos)

        img = p.getCameraImage(224, 224)[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('ant_data/{}.jpg'.format(i), gray)
        i += 1
        time.sleep(0.1)
        p.stepSimulation()

    p.disconnect()


makeDataset('snake.txt', numSamples=10)
