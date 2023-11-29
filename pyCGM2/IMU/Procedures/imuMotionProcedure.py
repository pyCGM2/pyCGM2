import numpy as np
from pyCGM2.Model import frame
from pyCGM2.IMU import imu

class ImuMotionProcedure(object):
    """
    Base class for procedures to construct motion from IMU data.
    """
    def __init__(self):
        pass

class QuaternionMotionProcedure(ImuMotionProcedure):
    """
    Construct motion from quaternions.

    Args:
        quaternions (np.ndarray): Array of quaternions.
    """
    def __init__(self,quaternions:np.ndarray):
        """Initializes the QuaternionMotionProcedure with quaternions data."""
        super(QuaternionMotionProcedure, self).__init__()
        self.m_quaternions =  quaternions


    def compute(self,imuInstance:imu.Imu):
        """
        Compute the procedure to create motion frames from quaternions.

        Args:
            imuInstance (imu.Imu): An IMU instance to which the motion frames will be added.
        """
        for i in range(0,self.m_quaternions.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromQuaternion(self.m_quaternions[i,:])
            imuInstance.m_motion.append(imuFrame)

class GlobalAngleMotionProcedure(ImuMotionProcedure):
    """
    Construct motion from global angles (equivalent to angle-axis representation).

    Args:
        globalAngle (np.ndarray): Array of global angles.
    """
    def __init__(self,globalAngle):
        """Initializes the GlobalAngleMotionProcedure with global angles data."""

        super(GlobalAngleMotionProcedure, self).__init__()
        self.m_globalAngle =  globalAngle


    def compute(self,imuInstance:imu.Imu):
        """
        Compute the procedure to create motion frames from global angles.

        Args:
            imuInstance (imu.Imu): An IMU instance to which the motion frames will be added.
        """
        for i in range(0,self.m_globalAngle.shape[0]):
            imuFrame=frame.Frame()
            imuFrame.constructFromAnglesAxis(self.m_globalAngle[i,:])
            imuInstance.m_motion.append(imuFrame)



class RealignedMotionProcedure(ImuMotionProcedure):
    """
    Realigned IMU motion relative to the first frame.
    """
    def __init__(self):
        super(RealignedMotionProcedure, self).__init__()


    def compute(self,imuInstance:imu.Imu):
        """
        Compute the procedure to realign IMU motion frames relative to the first frame.

        Args:
            imuInstance (imu.Imu): An IMU instance whose motion frames will be realigned.
        """
        trial_initial_DCM = np.linalg.inv(imuInstance.m_motion[0].getRotation())

        for i in range(0,len(imuInstance.getMotion())):
            rot = np.dot(trial_initial_DCM,imuInstance.m_motion[i].getRotation())
            imuInstance.m_motion[i].setRotation(rot) 





