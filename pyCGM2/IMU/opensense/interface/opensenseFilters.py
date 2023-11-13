
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuPlacerInterfaceProcedure
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuKinematicFitterProcedure


class opensenseInterfaceImuPlacerFilter(object):
    """filter to tor run the opensense IMU placer

        Args:
            procedure (opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure): the opensense IMU placer procedure
    """
    def __init__(self, procedure:opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure):
        self.m_procedure = procedure

    def run(self):
        """run the filter"""
        self.m_procedure.run()

    def getCalibratedOsimName(self):
        """return the name of the calibrated osim file"""
        return self.m_procedure.m_osim_calibrated


class opensenseInterfaceImuInverseKinematicFilter(object):
    """filter to tor run the opensense IMU Inverse kinematics

        Args:
            procedure (opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure): the opensense IMU inverse kinematics procedure
    """
    def __init__(self,procedure:opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()