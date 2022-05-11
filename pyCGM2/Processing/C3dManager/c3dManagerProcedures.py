# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module aims to organize the c3d trials according to computational objectives,
ie computation of spatio-temporal, kinematic, kinetics or emg parameters.
In practice, a unique c3d set (`UniqueC3dSetProcedure`) or separate c3d set (`DistinctC3dSetProcedure`)
can be considered whether you want to use the same c3d set or a different c3d set for acheiving objectives

The `C3dManager` instance is final object instance built from the `C3dManagerFilter`.
The `C3dManager` is a structure listing for each objectives,  the Btk.Acquisition instances and their associated
c3d filenames

"""


from pyCGM2.Tools import btkTools
from pyCGM2.Processing.C3dManager import c3dManagerFilters


class C3dManagerProcedure(object):
    def __init__(self):
        pass

class UniqueBtkAcqSetProcedure(C3dManagerProcedure):
    """the same combinaison (btk.Acquisition/c3d filenames) is used for all
    computational objectives

    Args:
        data_path (str): folder path
        fileLst (list): c3d filenames
        acqs (list): btk.Acquisition instances

    **Warning:**

    btk.Acquisition instances must match filenames.
    The first btk.acquisition instance must be the btk.acquisition from the first c3d filename.


    """


    def __init__(self, data_path, fileLst,acqs):
        super(UniqueBtkAcqSetProcedure,self).__init__()
        self.m_files = fileLst
        self.m_data_path = data_path
        self.m_acqs = acqs



    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):
        """disseminate the combinaison (btk.Acquisition/c3d filenames)

        Args:
            c3dManager (pyCGM2.Processing.c3dManager.C3dManager): a `c3dManager` instance.
            spatioTempFlag (bool): enable populating the `spatioTemporal` attribute of the  `c3dManager` instance
            kinematicFlag (bool): enable populating the `kinematic` attribute of the  `c3dManager` instance
            kineticFlag (bool): enable populating the `kinetic` attribute of the  `c3dManager` instance
            emgFlag (bool): enable populating the `emg` attribute of the  `c3dManager` instance

        """


        #---spatioTemporalTrials
        if spatioTempFlag:
            c3dManager.spatioTemporal["Acqs"] = self.m_acqs
            c3dManager.spatioTemporal["Filenames"] = self.m_files

        # ----kinematic trials---
        if kinematicFlag:
            c3dManager.kinematic["Acqs"] = self.m_acqs
            c3dManager.kinematic["Filenames"] = self.m_files

        #---kinetic Trials--- ( check if kinetic events)
        if kineticFlag:
            c3dManager.kinetic["Acqs"], c3dManager.kinetic["Filenames"], c3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files,acqs = self.m_acqs)

        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"] =self.m_acqs
            c3dManager.emg["Filenames"] = self.m_files



class UniqueC3dSetProcedure(C3dManagerProcedure):
    """the same c3d filenames is used for all computational objectives

    Args:
        data_path (str): folder path
        fileLst (list): c3d filenames
    """


    def __init__(self, data_path, fileLst):
        super(UniqueC3dSetProcedure,self).__init__()
        self.m_files = fileLst
        self.m_data_path = data_path



    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):
        """disseminate c3d filenames

        Args:
            c3dManager (pyCGM2.Processing.c3dManager.C3dManager): a `c3dManager` instance.
            spatioTempFlag (bool): enable populating the `spatioTemporal` attribute of the  `c3dManager` instance
            kinematicFlag (bool): enable populating the `kinematic` attribute of the  `c3dManager` instance
            kineticFlag (bool): enable populating the `kinetic` attribute of the  `c3dManager` instance
            emgFlag (bool): enable populating the `emg` attribute of the  `c3dManager` instance

        """


        #---spatioTemporalTrials
        if spatioTempFlag:
            c3dManager.spatioTemporal["Acqs"],c3dManager.spatioTemporal["Filenames"] = btkTools.buildTrials(self.m_data_path,self.m_files)


        # ----kinematic trials---
        if kinematicFlag:
            c3dManager.kinematic["Acqs"],c3dManager.kinematic["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)

        #---kinetic Trials--- ( check if kinetic events)
        if kineticFlag:
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],c3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)


class DistinctC3dSetProcedure(C3dManagerProcedure):
    """Distinct c3d sets are for each computational objectives

    Args:
        data_path (str): folder path
        stp_fileLst (list): c3d filenames for the spatioTemporal computation
        kinematic_fileLst (list): c3d filenames for the kinematic computation
        kinetic_fileLst (list): c3d filenames for the kinetics computation
        emg_fileLst (list): c3d filenames for the emg computation
    """

    def __init__(self, data_path, stp_fileLst, kinematic_fileLst, kinetic_fileLst, emg_fileLst):
        super(DistinctC3dSetProcedure,self).__init__()
        self.m_data_path = data_path

        self.m_files_stp = stp_fileLst
        self.m_files_kinematic = kinematic_fileLst
        self.m_files_kinetic = kinetic_fileLst
        self.m_files_emg = emg_fileLst

    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):
        """disseminate c3d sets

        Args:
            c3dManager (pyCGM2.Processing.c3dManager.C3dManager): a `c3dManager` instance.
            spatioTempFlag (bool): enable populating the `spatioTemporal` attribute of the  `c3dManager` instance
            kinematicFlag (bool): enable populating the `kinematic` attribute of the  `c3dManager` instance
            kineticFlag (bool): enable populating the `kinetic` attribute of the  `c3dManager` instance
            emgFlag (bool): enable populating the `emg` attribute of the  `c3dManager` instance

        """

        #---spatioTemporalTrials
        if spatioTempFlag:
            c3dManager.spatioTemporal["Acqs"],c3dManager.spatioTemporal["Filenames"] = btkTools.buildTrials(self.m_data_path,self.m_files_stp)


        # ----kinematic trials---
        if kinematicFlag:
            c3dManager.kinematic["Acqs"],c3dManager.kinematic["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_kinematic)

        #---kinetic Trials--- ( check if kinetic events)
        if kineticFlag:
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],c3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files_kinetic)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_emg)
