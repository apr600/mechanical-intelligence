import time
import numpy as np
import math

import pybullet as p
import pybullet_data as pd

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 7#11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class FrankaEnv(object):
    def __init__(self, render=True, ts=1./60., offset=[0.,0.,0.],
                 x0 = [0.,0.,0.]):

        self.bullet_client = p
        # Set up Pybullet Environment
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(ts)
        p.setGravity(0,-9.8,0)
        p.setPhysicsEngineParameter()#allowedCcdPenetration=0.0)
        
        # Set up Franka
        self.offset = np.array(offset)
        orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.robot = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array(x0)+self.offset, orn, useFixedBase=True, flags=flags)
        self.ee_ind = pandaEndEffectorIndex
    
        index = 0
        self.numJoints = self.bullet_client.getNumJoints(self.robot)
        for j in range(self.bullet_client.getNumJoints(self.robot)):
            self.bullet_client.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.robot, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):  
                self.bullet_client.resetJointState(self.robot, j, jointPositions[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.robot, j, jointPositions[index]) 
                index=index+1

        self.curr_pos, self.curr_orn, localInertialFramePosition, localInertialFrameOrientation,worldLinkFramePosition, worldLinkFrameOrientation= self.bullet_client.getLinkState(self.robot, 7)

        # Set Camera Properties
        self.camTargetPos = np.array([0, 0, 0])
        self.camUpVector = [1, 0, 0]
        self.projMat = self.bullet_client.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=3.1)
        
        self.t = 0.

        # Set up table/objects in Environment
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        tableID = self.bullet_client.loadURDF("/home/enrique/Documents/research/projects/LearningGenerativeSensorModel/camera_sim/franka/urdf/table_plain.urdf", np.array( [0, -.75, -0.6])+self.offset, [-0.5, -0.5, -0.5, 0.5], flags=flags)
        self.bullet_client.changeVisualShape(tableID, linkIndex=-1, rgbaColor=[0.87, .721, .529, 1])

        # self.bullet_client.changeVisualShape(tableID, linkIndex=-1, rgbaColor=[1.,1.,1,1.])

        # self.objID = self.bullet_client.loadURDF("duck_vhacd.urdf",np.array( [-0.1, -0.1, -0.4])+self.offset,  [-0.5, -0.5, -0.5, 0.5], flags=flags, useFixedBase=True)
        # print(self.bullet_client.getBasePositionAndOrientation(self.objID))

        self.objID = self.bullet_client.loadURDF("cube_small.urdf",np.array( [-0.1, -0.1, -0.4])+self.offset, flags=flags, useFixedBase=True)
        self.bullet_client.changeVisualShape(self.objID, linkIndex=-1, rgbaColor=[1,0,0,1.])


        self.obj2ID = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0.1, -0.1, -0.6])+self.offset, flags=flags, useFixedBase=True)
        self.bullet_client.changeVisualShape(self.obj2ID, linkIndex=-1, rgbaColor=[0,0,1,1.])

        input("Press Enter to start sim")
        
    def reset(self):
        pass
      
    def step(self, pos, orn):
        # Move robot to target position and orientation
        jointPoses = self.bullet_client.calculateInverseKinematics(self.robot,self.ee_ind, pos, orn, ll, ul, jr, rp, maxNumIterations=5)
        # for i in range(pandaNumDofs):
        #     self.bullet_client.setJointMotorControl2(self.robot, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)

        # Update Camera view
        # endPos, endOrn,localInertialFramePosition, localInertialFrameOrientation,worldLinkFramePosition, worldLinkFrameOrientation= self.bullet_client.getLinkState(self.robot, 7)
        index = 0
        for j in range(len(jointPoses)):
            # p.resetJointState(self.robot,j,jointPoses[j])
            info = self.bullet_client.getJointInfo(self.robot, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):  
                self.bullet_client.resetJointState(self.robot, j, jointPoses[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.robot, j, jointPoses[index]) 
                index=index+1


        # endPos = pos
        # endOrn = orn

        # rotMat = np.array(self.bullet_client.getMatrixFromQuaternion(endOrn)).reshape((3,3))

        # camPos = np.array(endPos)+np.dot(rotMat,np.array([0,0.,.1])) 
        # targetPos = np.array(endPos)+np.dot(rotMat, np.array([0,0,0.5]))
        # viewMatrix = self.bullet_client.computeViewMatrix(cameraEyePosition=camPos, cameraTargetPosition=targetPos, cameraUpVector = self.camUpVector)
        # width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=224, height=224, viewMatrix=viewMatrix,projectionMatrix=self.projMat)
        # self.cam_img = rgbImg.copy()

        # self.curr_pos = endPos
        # self.curr_orn = endOrn
        p.stepSimulation()
        pass  

    # def get_ee_state(self):
    #     eePos, eeOrn, _, _, _, _= self.bullet_client.getLinkState(self.robot, 7)
    #     eeOrn = self.bullet_client.getEulerFromQuaternion(eeOrn)
    #     return eePos, eeOrn

if __name__ == "__main__":
    # # Initialize Franka Robot
    timeStep=1./60.
    offset = [0.,0.,0.]
    env = FrankaEnv(render=True, ts=timeStep,offset=offset)
