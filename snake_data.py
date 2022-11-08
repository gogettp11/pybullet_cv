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
snake = p.loadURDF('snake.urdf')
numJoints = p.getNumJoints(snake)


def makeDataset(filename, numSamples, stickHeadToFloor=False):
    with open(filename, 'w') as f:
        f.truncate(0)

    i = 0
    while i < numSamples:
        # randomize initial orientation for the first segment
        if not stickHeadToFloor:
            initShift = np.random.rand(1)
            initAngle = np.random.rand(1) * 6.28
            initAngle *= random.choice((-1, 1))
            p.resetBasePositionAndOrientation(snake,
                                              [initShift, 0, 0.25],
                                              p.getQuaternionFromEuler([initAngle, 1.57, 0]))
        # pose the other two segments
        jointPositions = np.random.rand(numJoints)
        for joint in range(numJoints):
            jointPositions[joint] *= random.choice((-1, 1))
            p.resetJointState(snake, joint, jointPositions[joint])

        decimals = 4
        with open(filename, 'a') as f:
            line = 'id: {} joints: '.format(i)
            for joint in jointPositions:
                line += str(round(joint, decimals)) + ' '
            f.write(line + '\n')

        snakePos, _ = p.getBasePositionAndOrientation(snake)
        p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraPitch=-89, cameraYaw=0,
                                     cameraTargetPosition=snakePos)

        img = p.getCameraImage(224, 224)[2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('snake_data/{}.jpg'.format(i), gray)
        i += 1
        time.sleep(0.1)
        p.stepSimulation()

    p.disconnect()


makeDataset('snake.txt', numSamples=10, stickHeadToFloor=False)
