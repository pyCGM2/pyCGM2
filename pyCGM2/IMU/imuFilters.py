
from pyCGM2.IMU.Procedures.imuMotionProcedure import ImuMotionProcedure
from pyCGM2.IMU.Procedures.relativeImuAngleProcedures import RelativeImuAnglesProcedure
from pyCGM2.IMU.Procedures.imuReaderProcedures import ImuReaderProcedure
from pyCGM2.IMU import imu

from typing import List, Tuple, Dict, Optional,Union

class ImuReaderFilter(object):
    """
    Filter for reading IMU data using a specified reading procedure.

    Args:
        procedure (ImuReaderProcedure): The reading procedure to be used for reading IMU data.
    """
    def __init__(self,procedure:ImuReaderProcedure):
        """Initializes the ImuReaderFilter with the specified reading procedure."""
        self.m_procedure=procedure
  
    def run(self):
        """
        Executes the filter to read IMU data based on the specified procedure.

        Returns:
            Imu: An instance of the Imu class with the read data.
        """
        return self.m_procedure.read()

class ImuMotionFilter(object):
    """
    Filter for generating IMU motion from an IMU instance using a specified motion procedure.

    Args:
        imuInstance (imu.Imu): The IMU instance to generate motion for.
        procedure (ImuMotionProcedure): The motion procedure to generate motion data.
    """
    def __init__(self,imuInstance:imu.Imu,procedure:ImuMotionProcedure):
        """Initializes the ImuMotionFilter with the specified IMU instance and motion procedure."""
        self.m_imu=imuInstance
        self.m_procedure=procedure
  
    def run(self):
        """Executes the filter to generate IMU motion based on the specified procedure."""
        self.m_procedure.compute(self.m_imu)


class ImuRelativeAnglesFilter(object):
    """
    Filter for calculating the relative angles between two IMUs.

    Args:
        imuInstance1 (imu.Imu): The first IMU instance.
        imuInstance2 (imu.Imu): The second IMU instance.
        procedure (RelativeImuAnglesProcedure): The procedure to compute relative angles between the IMUs.
    """
    def __init__(self,imuInstance1:imu.Imu,imuInstance2:imu.Imu,procedure:RelativeImuAnglesProcedure):
        """Initializes the ImuRelativeAnglesFilter with the specified IMU instances and relative angles procedure."""

        self.m_imu1=imuInstance1
        self.m_imu2=imuInstance2
        self.m_procedure=procedure

    def run(self):
        """
        Executes the filter to calculate relative angles between the two IMU instances.

        Returns:
            np.ndarray: The calculated angles between the two IMUs.
        """
        out = self.m_procedure.compute(self.m_imu1,self.m_imu2)
        return out