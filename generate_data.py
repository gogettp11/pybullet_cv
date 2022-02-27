import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random
import csv
import os
import shutil

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane = p.loadURDF('plane.urdf')
table = p.loadURDF('table/table.urdf')
robotId = p.loadSDF('kuka_iiwa/kuka_with_gripper2.sdf')
robot = robotId[0]
numJoints = p.getNumJoints(robot)
p.resetBasePositionAndOrientation(robot, [0, 0, 0.7], [0,0,0,1])

robotPos = p.getBasePositionAndOrientation(robot)[0]
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=180, cameraPitch=-41,
                             cameraTargetPosition=robotPos)


def randomColor():
    color = np.random.rand(4)
    color[3] = 1
    return color


def createDataset(filename, numSamples):

    with open(filename, 'w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        shutil.rmtree("images")
        os.makedirs("images")
        decimals = 4
        for i in range(numSamples):

            p.changeVisualShape(plane, -1, rgbaColor=randomColor())
            p.changeVisualShape(table, -1, rgbaColor=randomColor())

            color = randomColor()
            jointPositions = np.random.rand(numJoints) * 2.

            for joint in range(numJoints):
                p.changeVisualShape(robot, joint, rgbaColor=color)

                jointPositions[joint] *= random.choice((-1, 1))
                p.resetJointState(robot, joint, jointPositions[joint])

            csv_writer.writerow(np.around(jointPositions, decimals))

            w, h, rgba, depth, mask = p.getCameraImage(224, 224)
            rgba = rgba[:, :, 0:3] #rgba to rgb
            depth = np.expand_dims(depth, axis=2)
            depth_img = np.concatenate((rgba, depth), axis=2)
            cv2.imwrite(f'images/{i}.jpg', depth_img)

            p.stepSimulation()
    p.disconnect()


createDataset('joints.csv', 10000)
