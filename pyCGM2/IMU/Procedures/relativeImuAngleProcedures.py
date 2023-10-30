# -*- coding: utf-8 -*-
import numpy as np
from  pyCGM2.Math import euler


class AbstractRelativeImuAnglesProcedure(object):
    def __init__(self):
        self.m_fixEuler =  True
        pass

class RelativeAnglesProcedure(AbstractRelativeImuAnglesProcedure):
    def __init__(self,representation="Euler", eulerSequence="XYZ"):
        super(RelativeAnglesProcedure, self).__init__()

        self.m_representation = representation
        self.m_eulerSequence = eulerSequence


    def run(self, imuInstance1, imuInstance2):

        motion1 = imuInstance1.getMotion()
        motion2 = imuInstance2.getMotion()


        nFrames = min([len(motion1), len(motion2)])

        jointValues = np.zeros((nFrames,3))
        for i in range (0, nFrames):

            Rprox = motion1[i].getRotation()
            Rdist = motion2[i].getRotation()
            Rrelative= np.dot(Rprox.T, Rdist)

            if self.m_representation == "Euler":
                if self.m_eulerSequence == "ZYX":
                    Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative,similarOrder=False)    
                elif self.m_eulerSequence == "XZY":
                    Euler1,Euler2,Euler3 = euler.euler_xzy(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "YXZ":
                    Euler1,Euler2,Euler3 = euler.euler_yxz(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "YZX":
                    Euler1,Euler2,Euler3 = euler.euler_yzx(Rrelative,similarOrder=False)
                elif self.m_eulerSequence == "ZXY":
                    Euler1,Euler2,Euler3 = euler.euler_zxy(Rrelative,similarOrder=False)
                elif self.m_eulerSequence  == "XYZ":
                    Euler1,Euler2,Euler3 = euler.euler_xyz(Rrelative,similarOrder=False)
                else:
                    raise Exception("[pyCGM2] joint sequence unknown ")
                
                jointValues[i,0] = Euler1
                jointValues[i,1] = Euler2
                jointValues[i,2] = Euler3
           

        jointFinalValues = jointValues 

        if self.m_fixEuler:
            dest = np.array([0,0,0])
            for i in range (0, nFrames):
                jointFinalValues[i,:] = euler.wrapEulerTo(jointFinalValues[i,:], dest)
                dest = jointFinalValues[i,:]

        return jointFinalValues

        
