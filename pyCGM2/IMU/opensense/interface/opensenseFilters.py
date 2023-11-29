
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuPlacerInterfaceProcedure
from pyCGM2.IMU.opensense.interface.procedures import opensenseImuKinematicFitterProcedure


class opensenseInterfaceImuPlacerFilter(object):
    """
    Filter to run the OpenSense IMU Placer.

    This filter encapsulates the functionality of the OpenSense IMU Placer procedure and provides a simple interface 
    to run the procedure and get the results.

    Args:
        procedure (opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure): The OpenSense IMU placer procedure instance.
    """

    def __init__(self, procedure:opensenseImuPlacerInterfaceProcedure.ImuPlacerXMLProcedure):
        """Initializes the filter with the specified OpenSense IMU Placer procedure."""
        self.m_procedure = procedure

    def run(self):
        """
        Run the OpenSense IMU Placer procedure.
        """
        self.m_procedure.run()

    def getCalibratedOsimName(self):
        """
        Get the name of the calibrated OpenSim model file.

        Returns:
            str: The name of the calibrated OpenSim model file.
        """
        return self.m_procedure.m_osim_calibrated


class opensenseInterfaceImuInverseKinematicFilter(object):
    """
    Filter to run the OpenSense IMU Inverse Kinematics.

    This filter encapsulates the functionality of the OpenSense IMU Inverse Kinematics procedure and provides a simple 
    interface to run the procedure.

    Args:
        procedure (opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure): The OpenSense IMU inverse kinematics procedure instance.
    """
    def __init__(self,procedure:opensenseImuKinematicFitterProcedure.ImuInverseKinematicXMLProcedure):
        """Initializes the filter with the specified OpenSense IMU Inverse Kinematics procedure."""

        self.m_procedure = procedure

    def run(self):
        """
        Run the OpenSense IMU Inverse Kinematics procedure.
        """
        self.m_procedure.run()