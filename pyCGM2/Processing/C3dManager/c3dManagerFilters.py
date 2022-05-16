# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to organize the c3d trials according to computational objectives,
ie computation of spatio-temporal, kinematic, kinetics or emg parameters.

The implementation is based on a *builder pattern*.
The filter `C3dManagerFilter` calls a procedure, then return a `C3dManager` instance, a
ie ,  a structure listing both Btk.Acquisition instances and their associated filename
according the computational objectives

.. code-block:: python

    c3dmanagerProcedure = c3dManagerProcedures.DistinctC3dSetProcedure(
        DATA_PATH, iPstFilenames, iKinematicFilenames, iKineticFilenames, iEmgFilenames)

    cmf = c3dManagerFilters.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableKinematic(True)
    cmf.enableKinetic(True)
    cmf.enableEmg(True)
    trialManager = cmf.generate()

"""

from pyCGM2.Tools import btkTools


class C3dManager(object):
    """ A `c3d manager` instance is a structure listing btk.Acquisition instances and
    the associated filenames for the 4 computational objectives
    (spatio-temporal, kinematic, kinetics or emg computation)

    """
    def __init__ (self):
        self.spatioTemporal={"Acqs":None , "Filenames":None}
        self.kinematic={"Acqs":None , "Filenames":None}
        self.kineticFlag = False
        self.kinetic={"Acqs": None, "Filenames":None }
        self.emg={"Acqs":None , "Filenames":None}


class C3dManagerFilter(object):
    """
    pyCGM2 Filter used for disseminate c3d trial set(s)

    Args:
        procedure ( pyCGM2.Processing.C3dManager.c3dManagerProcedures.C3dManagerProcedure): a C3dManagerProcedure instance
    """

    def __init__(self,procedure):

        self.m_procedure = procedure
        self.m_spatioTempFlag = True
        self.m_kinematicFlag = True
        self.m_kineticFlag = True
        self.m_emgFlag = True

    def enableSpatioTemporal(self, boolean):
        """enable spatio-temporal computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_spatioTempFlag = boolean

    def enableKinematic(self, boolean):
        """enable kinematic computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_kinematicFlag = boolean

    def enableKinetic(self, boolean):
        """enable kinetic computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_kineticFlag = boolean

    def enableEmg(self, boolean):
        """enable emg computation

        Args:
            boolean (bool): boolean flag

        """
        self.m_emgFlag = boolean



    def generate(self):
        """ disseminate c3d trials according to the given Procedure
        """

        c3dManager = C3dManager()


        self.m_procedure.generate(c3dManager,self.m_spatioTempFlag, self.m_kinematicFlag, self.m_kineticFlag, self.m_emgFlag)

        return c3dManager
