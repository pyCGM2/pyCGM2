"""
This module provides procedures for managing C3D files used in biomechanical analyses. 
It includes various procedures to organize and disseminate C3D data sets based on different 
computational objectives such as spatio-temporal analysis, kinematics, kinetics, and EMG. 
The procedures are designed to work with the `C3dManagerFilter` to facilitate the selection 
and categorization of C3D files for subsequent analyses.
"""



from pyCGM2.Tools import btkTools
import btk

from typing import List, Tuple, Dict, Optional,Union,Any

from pyCGM2.Processing.C3dManager.c3dManager import C3dManager

class C3dManagerProcedure(object):
    """
    Base class for C3D management procedures.

    This class serves as a foundational structure for specific procedures that organize C3D trials.
    Derived classes should implement specific strategies for managing C3D files.
    """
    def __init__(self):
        pass

class UniqueBtkAcqSetProcedure(C3dManagerProcedure):
    """
    A procedure where the same combination of btk.Acquisition instances and C3D filenames 
    is used for all computational objectives.

    Args:
        data_path (str): Directory path of C3D files.
        fileLst (List[str]): List of C3D filenames.
        acqs (List[btk.btkAcquisition]): List of btk.Acquisition instances.

    Warning:
        btk.btkAcquisition instances must match the filenames provided.
    """
    

    def __init__(self, data_path:str, fileLst:List[str],acqs:List[btk.btkAcquisition]):
        super(UniqueBtkAcqSetProcedure,self).__init__()
        self.m_files = fileLst
        self.m_data_path = data_path
        self.m_acqs = acqs



    def generate(self,c3dManager:C3dManager,spatioTempFlag:bool,kinematicFlag:bool,kineticFlag:bool,emgFlag:bool,muscleGeometryFlag:bool,muscleDynamicFlag:bool ):
        """
        Disseminates a combination of btk.btkAcquisition instances and C3D filenames across different computational categories in the C3dManager instance.

        Args:
            c3dManager (C3dManager): The C3dManager instance to be populated.
            spatioTempFlag (bool): If True, populates the spatio-temporal category.
            kinematicFlag (bool): If True, populates the kinematic category.
            kineticFlag (bool): If True, populates the kinetic category.
            emgFlag (bool): If True, populates the EMG category.
            muscleGeometryFlag (bool): If True, populates the muscle geometry category.
            muscleDynamicFlag (bool): If True, populates the muscle dynamic category.
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
    """
    Procedure using the same C3D filenames for all computational objectives.

    Args:
        data_path (str): Path to the folder containing C3D files.
        fileLst (List[str]): List of C3D filenames to be used across all computational categories.
    """
    
    def __init__(self, data_path:str, fileLst:List[str]):
        super(UniqueC3dSetProcedure,self).__init__()
        self.m_files = fileLst
        self.m_data_path = data_path



    def generate(self,c3dManager:C3dManager,spatioTempFlag:bool,kinematicFlag:bool,kineticFlag:bool,emgFlag:bool,muscleGeometryFlag:bool,muscleDynamicFlag:bool ):
        """
        Disseminates a combination of btk.btkAcquisition instances and C3D filenames across different computational categories in the C3dManager instance.

        Args:
            c3dManager (C3dManager): The C3dManager instance to be populated.
            spatioTempFlag (bool): If True, populates the spatio-temporal category.
            kinematicFlag (bool): If True, populates the kinematic category.
            kineticFlag (bool): If True, populates the kinetic category.
            emgFlag (bool): If True, populates the EMG category.
            muscleGeometryFlag (bool): If True, populates the muscle geometry category.
            muscleDynamicFlag (bool): If True, populates the muscle dynamic category.
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

        #----muscleGeometryTrials
        if muscleGeometryFlag:
            c3dManager.muscleGeometry["Acqs"],c3dManager.muscleGeometry["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)

        #----muscleDynamicTrials
        if muscleDynamicFlag:
            c3dManager.muscleDynamic["Acqs"],c3dManager.muscleDynamic["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)

class DistinctC3dSetProcedure(C3dManagerProcedure):
    """
    Procedure using distinct C3D sets for each computational objective.

    Args:
        data_path (str): Path to the folder containing C3D files.
        stp_fileLst (List[str]): C3D filenames for spatio-temporal computation.
        kinematic_fileLst (List[str]): C3D filenames for kinematic computation.
        kinetic_fileLst (List[str]): C3D filenames for kinetic computation.
        emg_fileLst (List[str]): C3D filenames for EMG computation.
        muscleGeometry_fileLst (List[str]): C3D filenames for muscle geometry computation.
        muscleDynamic_fileLst (List[str]): C3D filenames for muscle dynamic computation.
    """
    

    def __init__(self, data_path, stp_fileLst, kinematic_fileLst, kinetic_fileLst, emg_fileLst, 
                muscleGeometry_fileLst, muscleDynamic_fileLst):
        super(DistinctC3dSetProcedure,self).__init__()
        self.m_data_path = data_path

        self.m_files_stp = stp_fileLst
        self.m_files_kinematic = kinematic_fileLst
        self.m_files_kinetic = kinetic_fileLst
        self.m_files_emg = emg_fileLst
        self.m_files_muscleGeometry = muscleGeometry_fileLst
        self.m_files_muscleDynamic = muscleDynamic_fileLst

    def generate(self,c3dManager:C3dManager,spatioTempFlag:bool,kinematicFlag:bool,kineticFlag:bool,emgFlag:bool,muscleGeometryFlag:bool,muscleDynamicFlag:bool ):
        """
        Disseminates a combination of btk.btkAcquisition instances and C3D filenames across different computational categories in the C3dManager instance.

        Args:
            c3dManager (C3dManager): The C3dManager instance to be populated.
            spatioTempFlag (bool): If True, populates the spatio-temporal category.
            kinematicFlag (bool): If True, populates the kinematic category.
            kineticFlag (bool): If True, populates the kinetic category.
            emgFlag (bool): If True, populates the EMG category.
            muscleGeometryFlag (bool): If True, populates the muscle geometry category.
            muscleDynamicFlag (bool): If True, populates the muscle dynamic category.
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

        #----muscleGeometryTrials
        if muscleGeometryFlag:
            c3dManager.muscleGeometry["Acqs"],c3dManager.muscleGeometry["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_muscleGeometry)

        #----muscleDynamicTrials
        if muscleDynamicFlag:
            c3dManager.muscleDynamic["Acqs"],c3dManager.muscleDynamic["Filenames"], c3dManager.muscleDynamicFlag = btkTools.automaticKineticDetection(self.m_data_path,self.m_files_muscleDynamic)
            