"""
This module is designed to manage and organize C3D trials based on different computational objectives, such as spatio-temporal, kinematic, kinetic, and EMG parameter computations.

The implementation follows a builder pattern, where the `C3dManagerFilter` is used to create and configure a `C3dManager` instance. This instance acts as a structured repository that lists btk.Acquisition instances and their associated filenames, organized according to the computational objectives.

Example usage:

.. code-block:: python

    c3dmanagerProcedure = c3dManagerProcedures.DistinctC3dSetProcedure(
        DATA_PATH, iPstFilenames, iKinematicFilenames, iKineticFilenames, iEmgFilenames)

    cmf = c3dManagerFilters.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableKinematic(True)
    cmf.enableKinetic(True)
    cmf.enableEmg(True)
    trialManager = cmf.generate()
"""
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Processing.C3dManager.c3dManagerProcedures import C3dManagerProcedure
from pyCGM2.Processing.C3dManager import c3dManager

from typing import List, Tuple, Dict, Optional,Union,Any

class C3dManagerFilter(object):
    """
    A filter for managing and disseminating sets of C3D trials.

    This filter is used to organize C3D trials into distinct sets based on computational objectives like spatio-temporal, kinematic, kinetic, and EMG parameters.

    Args:
        procedure (C3dManagerProcedure): An instance of a C3dManagerProcedure.
    """

    def __init__(self,procedure:C3dManagerProcedure):

        self.m_procedure = procedure
        self.m_spatioTempFlag = True
        self.m_kinematicFlag = True
        self.m_kineticFlag = True
        self.m_emgFlag = True
        self.m_muscleGeometryFlag = False
        self.m_muscleDynamicFlag = False

    def enableSpatioTemporal(self, boolean:bool):
        """ Enable/disable spatio-temporal computation. 

        Args:
            boolean (bool): boolean flag

        """
        self.m_spatioTempFlag = boolean

    def enableKinematic(self, boolean:bool):
        """enable/disable kinematic computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_kinematicFlag = boolean

    def enableKinetic(self, boolean:bool):
        """enable/disable kinetic computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_kineticFlag = boolean

    def enableEmg(self, boolean:bool):
        """enable/disable emg computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_emgFlag = boolean


    def enableMuscleGeometry(self, boolean:bool):
        """enable/disable Muscle geometry computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_muscleGeometryFlag = boolean

    def enableMuscleDynamic(self, boolean:bool):
        """enable/disable muscle dynamics computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_muscleDynamicFlag = boolean



    def generate(self):
        """
        Generates and returns a C3dManager instance based on the specified procedure and the enabled computational objectives.

        This method disseminates the C3D trials into different categories such as spatio-temporal, kinematic, kinetic, EMG, muscle geometry and muscle  

        Returns:
            C3dManager: An instance of `C3dManager` that contains organized C3D trials as specified by the procedure and the enabled computational objectives.
        """

        c3dManagerInstance = c3dManager.C3dManager()
        

        self.m_procedure.generate(c3dManagerInstance,
                                self.m_spatioTempFlag,
                                self.m_kinematicFlag, 
                                self.m_kineticFlag, 
                                self.m_emgFlag,
                                self.m_muscleGeometryFlag,
                                self.m_muscleDynamicFlag)

        return c3dManagerInstance
