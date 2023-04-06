# -*- coding: utf-8 -*-
import numpy as np
from  pyCGM2.Math import euler
from pyCGM2.Model import frame

class AbstractRelativeImuAnglesProcedure(object):
    def __init__(self):
        self.m_fixEuler =  True
        pass

class BlueTridentsRelativeAnglesProcedure(AbstractRelativeImuAnglesProcedure):
    def __init__(self,representation="Euler", eulerSequence="XYZ"):
        super(BlueTridentsRelativeAnglesProcedure, self).__init__()

        self.m_representation = representation
        self.m_eulerSequence = eulerSequence





    def run(self, imuInstance1, imuInstance2):

        rotations1 = imuInstance1.m_data["Orientations"]["ViconGlobalAngles"]["RotationMatrix"]
        rotations2 = imuInstance2.m_data["Orientations"]["ViconGlobalAngles"]["RotationMatrix"]
 
        analogFrames = len(rotations1)


        jointValues = np.zeros((analogFrames,3))
        for i in range (0, analogFrames):
            Rprox = rotations1[i]
            Rdist = rotations2[i]
            Rrelative= np.dot(Rprox.T, Rdist)

            if self.m_representation == "Euler":
                if self.m_eulerSequence == "ZYX":
                    Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative,similarOrder=False)    

                    jointValues[i,0] = Euler1
                    jointValues[i,1] = Euler2
                    jointValues[i,2] = Euler3
                elif self.m_eulerSequence == "XZY":
                    Euler1,Euler2,Euler3 = euler.euler_xzy(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "YXZ":
                    Euler1,Euler2,Euler3 = euler.euler_yxz(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "YZX":
                    Euler1,Euler2,Euler3 = euler.euler_yzx(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "ZXY":
                    Euler1,Euler2,Euler3 = euler.euler_zxy(Rrelative,similarOrder=False)
                elif self.m_eulerSequence  == "ZYX":
                    Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative,similarOrder=False)
                else:
                    raise Exception("[pyCGM2] joint sequence unknown ")


            if  self.m_representation == "GlobalAngle":   

                quaternion = frame.getQuaternionFromMatrix(Rrelative)
                angleAxis = frame.angleAxisFromQuaternion(quaternion,toRad=True)
                
                jointValues[i,0] = np.linalg.norm(angleAxis)   

        jointFinalValues = np.rad2deg(jointValues)

        if self.m_representation == "Euler":
             if self.m_fixEuler:
                dest = np.deg2rad(np.array([0,0,0]))
                for i in range (0, analogFrames):
                    jointFinalValues[i,:] = euler.wrapEulerTo(np.deg2rad(jointFinalValues[i,:]), dest)
                    dest = jointFinalValues[i,:]
                jointFinalValues = np.rad2deg(jointValues)

        return jointFinalValues

        
