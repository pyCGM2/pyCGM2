import numpy as np
from pyCGM2.Model import frame
from pyCGM2.IMU import imu

class ImuMotionProcedure(object):
    def __init__(self):
        pass

class QuaternionMotionProcedure(ImuMotionProcedure):
    """ construct motion from quaternions
        Args:
            quaternions (np.ndarray): quaternions
    """
    def __init__(self,quaternions:np.ndarray):
        super(QuaternionMotionProcedure, self).__init__()
        self.m_quaternions =  quaternions


    def compute(self,imuInstance:imu.Imu):
        """compute the procedure

        Args:
            imuInstance (imu.Imu): an imu instance
        """
        for i in range(0,self.m_quaternions.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromQuaternion(self.m_quaternions[i,:])
            imuInstance.m_motion.append(imuFrame)

class GlobalAngleMotionProcedure(ImuMotionProcedure):
    """ construct motion from global angle (eq angle axis)
        Args:
            globalAngle (np.ndarray): global angle
    """
    def __init__(self,globalAngle):
        super(GlobalAngleMotionProcedure, self).__init__()
        self.m_globalAngle =  globalAngle


    def compute(self,imuInstance:imu.Imu):
        """compute the procedure

        Args:
            imuInstance (imu.Imu): an imu instance
        """
        for i in range(0,self.m_globalAngle.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromAnglesAxis(self.m_globalAngle[i,:])
            imuInstance.m_motion.append(imuFrame)



class RealignedMotionProcedure(ImuMotionProcedure):
    """realigne imu motion relative to the first frame
    """
    def __init__(self):
        super(RealignedMotionProcedure, self).__init__()


    def compute(self,imuInstance:imu.Imu):
        """compute the procedure

        Args:
            imuInstance (imu.Imu): an imu instance
        """
        trial_initial_DCM = np.linalg.inv(imuInstance.m_motion[0].getRotation())

        for i in range(0,len(imuInstance.getMotion())):
            rot = np.dot(trial_initial_DCM,imuInstance.m_motion[i].getRotation())
            imuInstance.m_motion[i].setRotation(rot) 





