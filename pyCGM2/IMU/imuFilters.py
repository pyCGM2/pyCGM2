
from pyCGM2.IMU.Procedures.imuMotionProcedure import ImuMotionProcedure
from pyCGM2.IMU.Procedures.relativeImuAngleProcedures import RelativeImuAnglesProcedure
from pyCGM2.IMU.Procedures.imuReaderProcedures import ImuReaderProcedure
from pyCGM2.IMU import imu

from typing import List, Tuple, Dict, Optional,Union

class ImuReaderFilter(object):
    """read imu data 

        Args:
            procedure (ImuReaderProcedure): a reader procedure
    """
    def __init__(self,procedure:ImuReaderProcedure):
        self.m_procedure=procedure
  
    def run(self):
        return self.m_procedure.read()

class ImuMotionFilter(object):
    """return the IMU motion

        Args:
            imuInstance (imu.Imu): an imu instance
            procedure (ImuMotionProcedure): a motion procedure
    """
    def __init__(self,imuInstance:imu.Imu,procedure:ImuMotionProcedure):
        self.m_imu=imuInstance
        self.m_procedure=procedure
  
    def run(self):
        """run the filter"""
        self.m_procedure.compute(self.m_imu)


class ImuRelativeAnglesFilter(object):
    """return the angle between 2 imus

        Args:
            imuInstance (imu.Imu): an imu instance
            procedure (RelativeImuAnglesProcedure): a relative Imu procedure
    """
    def __init__(self,imuInstance1:imu.Imu,imuInstance2:imu.Imu,procedure:RelativeImuAnglesProcedure):

        self.m_imu1=imuInstance1
        self.m_imu2=imuInstance2
        self.m_procedure=procedure

    def run(self):
        out = self.m_procedure.compute(self.m_imu1,self.m_imu2)
        return out