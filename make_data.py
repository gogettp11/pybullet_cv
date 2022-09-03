import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random
import os
from const import *

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF('plane.urdf')


def makeTurtleDataset(numSamples=10000):
    turtle = p.loadURDF('urdf/turtle.urdf', [0, 0, 0.25])
    makeDataset(turtle, 'turtle', numSamples, motionRange=0.3,
                camDistance=4, camPitch=-45, camYaw=0)


def makeAntDataset(numSamples=10000):
    ant = p.loadURDF('urdf/ant.urdf', [0, 0, 2])
    makeDataset(ant, 'ant', numSamples, motionRange=1,
                camDistance=4, camPitch=-60, camYaw=45)


def makeSnakeDataset(numSamples=10000):
    snake = p.loadURDF('urdf/snake.urdf', [0, 0, 0.25])
    makeDataset(snake, 'snake', numSamples, motionRange=2,
                camDistance=3, camPitch=-89, camYaw=0)


def makeManipulatorDataset(numSamples=10000):
    orn = p.getQuaternionFromEuler([0, -1.57, 0])
    manipulator = p.loadURDF('urdf/snake.urdf', [0, 0, 2.25], orn)
    makeDataset(manipulator, 'manipulator', numSamples, motionRange=2,
                camDistance=3, camPitch=0, camYaw=90)


def makeDataset(robot, robotName, numSamples, motionRange,
                camDistance, camPitch, camYaw):
    dir = f'{DATA_TRAIN}/{robotName}/'
    file_img = f'{dir}/images'
    file_joints = f'{dir}/joints'
    os.makedirs(os.path.dirname(dir), exist_ok=True) # create directory if it doesn't exist for txr file

    image_buf = []
    joint_buf = []
    temp_joints = []
    numJoints = p.getNumJoints(robot)
    i = 0
    for i in range(numSamples):
        # Randomize initial orientation for the first segment for snake
        if robotName == 'snake':
            initShift = np.random.rand(1)
            initAngle = np.random.rand(1) * 6.28
            initAngle *= random.choice((-1, 1))
            p.resetBasePositionAndOrientation(robot,
                                              [initShift, 0, 0.25],
                                              p.getQuaternionFromEuler([initAngle, 1.57, 0]))

        jointPositions = np.random.rand(numJoints) * motionRange
        for joint in range(numJoints):
            jointPositions[joint] *= random.choice((-1, 1))
            p.resetJointState(robot, joint, jointPositions[joint])

        decimals = 4
        for joint in jointPositions:
            temp_joints.append(round(joint, decimals))
        joint_buf.append(temp_joints)
        temp_joints.clear()

        robotPos, _ = p.getBasePositionAndOrientation(robot)
        p.resetDebugVisualizerCamera(cameraDistance=camDistance,
                                     cameraPitch=camPitch,
                                     cameraYaw=camYaw,
                                     cameraTargetPosition=robotPos)

        img = p.getCameraImage(224, 224)[2]
        img = np.reshape(img, (224, 224, 4))
        img = img[:, :, :3]
        image_buf.append(img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f'{dir_imgs}.bmp/{i}', img)

        time.sleep(0.1)
        p.stepSimulation()

    temp_imgs = np.array(image_buf)
    np.save(file_img, temp_imgs)
    temp_joints = np.array(joint_buf)
    np.save(file_joints, temp_joints)

    p.disconnect()


makeSnakeDataset(numSamples=30)
