import numpy as np
from pyCGM2.Model import frame


class AbstractImuMotionProcedure(object):
    def __init__(self):
        pass

class QuaternionMotionProcedure(AbstractImuMotionProcedure):
    def __init__(self,quaternions):
        super(QuaternionMotionProcedure, self).__init__()
        self.m_quaternions =  quaternions


    def compute(self,imuInstance):
        for i in range(0,self.m_quaternions.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromQuaternion(self.m_quaternions[i,:])
            imuInstance.m_motion.append(imuFrame)

class GlobalAngleMotionProcedure(AbstractImuMotionProcedure):
    def __init__(self,globalAngle):
        super(GlobalAngleMotionProcedure, self).__init__()
        self.m_globalAngle =  globalAngle


    def compute(self,imuInstance):

        for i in range(0,self.m_globalAngle.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromAnglesAxis(self.m_globalAngle[i,:])
            imuInstance.m_motion.append(imuFrame)



class RealignedMotionProcedure(AbstractImuMotionProcedure):
    def __init__(self):
        super(RealignedMotionProcedure, self).__init__()


    def compute(self,imuInstance):

        trial_initial_DCM = np.linalg.inv(imuInstance.m_motion[0].getRotation())

        for i in range(0,len(imuInstance.getMotion())):
            rot = np.dot(trial_initial_DCM,imuInstance.m_motion[i].getRotation())
            imuInstance.m_motion[i].setRotation(rot) 





