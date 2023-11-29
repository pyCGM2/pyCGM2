# -*- coding: utf-8 -*-
import numpy as np
from  pyCGM2.Math import euler
from pyCGM2.IMU import imu
from typing import List, Tuple, Dict, Optional,Union

class RelativeImuAnglesProcedure(object):
    """
    Base class for procedures to compute relative angles between two IMUs.

    This class serves as a foundation for implementing specific procedures that calculate the relative orientation 
    between two IMU sensors. It can be extended to use different methods of calculating these angles.
    """
    def __init__(self):
        self.m_fixEuler =  True
        pass

class RelativeAnglesProcedure(RelativeImuAnglesProcedure):

    """
    Procedure to compute angles between two IMUs.

    This class calculates the relative orientation between two IMUs using specified angle representation and Euler sequence.

    Args:
        representation (str, optional): Angle representation. Defaults to "Euler".
        eulerSequence (str, optional): Euler sequence to be used for angle calculation. Defaults to "XYZ".
    """

    def __init__(self,representation="Euler", eulerSequence="XYZ"):
        """Initializes the RelativeAnglesProcedure with the specified angle representation and Euler sequence."""
        super(RelativeAnglesProcedure, self).__init__()

        self.m_representation = representation
        self.m_eulerSequence = eulerSequence


    def compute(self, imuInstance1:imu.Imu, imuInstance2:imu.Imu)->np.ndarray:
        """
        Compute the relative angles between two IMU instances.

        The method calculates the relative orientation of one IMU with respect to another and converts it into angles based on the specified representation and Euler sequence.

        Args:
            imuInstance1 (imu.Imu): The first IMU instance.
            imuInstance2 (imu.Imu): The second IMU instance.

        Returns:
            np.ndarray: Array of angles representing the relative orientation between the two IMUs for each frame. The array shape is (nFrames, 3), where nFrames is the number of frames.
        """

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

        
