# -*- coding: utf-8 -*-
# pyCGM2
from pyCGM2.Tools import btkTools


class C3dManager(object):
    def __init__ (self):
        self.spatioTemporal={"Acqs":None , "Filenames":None}
        self.kinematic={"Acqs":None , "Filenames":None}
        self.kineticFlag = False
        self.kinetic={"Acqs": None, "Filenames":None }
        self.emg={"Acqs":None , "Filenames":None}



class UniqueBtkAcqSetProcedure(object):


    def __init__(self, data_path, fileLst,acqs):
        self.m_files = fileLst
        self.m_data_path = data_path
        self.m_acqs = acqs



    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):


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
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files,acqs = self.m_acqs)

        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"] =self.m_acqs
            c3dManager.emg["Filenames"] = self.m_files



class UniqueC3dSetProcedure(object):


    def __init__(self, data_path, fileLst):
        self.m_files = fileLst
        self.m_data_path = data_path



    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):


        #---spatioTemporalTrials
        if spatioTempFlag:
            c3dManager.spatioTemporal["Acqs"],c3dManager.spatioTemporal["Filenames"] = btkTools.buildTrials(self.m_data_path,self.m_files)


        # ----kinematic trials---
        if kinematicFlag:
            c3dManager.kinematic["Acqs"],c3dManager.kinematic["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)

        #---kinetic Trials--- ( check if kinetic events)
        if kineticFlag:
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)


class DistinctC3dSetProcedure(object):


    def __init__(self, data_path, stp_fileLst, kinematic_fileLst, kinetic_fileLst, emg_fileLst):

        self.m_data_path = data_path

        self.m_files_stp = stp_fileLst
        self.m_files_kinematic = kinematic_fileLst
        self.m_files_kinetic = kinetic_fileLst
        self.m_files_emg = emg_fileLst

    def generate(self,c3dManager,spatioTempFlag,kinematicFlag,kineticFlag,emgFlag):


        #---spatioTemporalTrials
        if spatioTempFlag:
            c3dManager.spatioTemporal["Acqs"],c3dManager.spatioTemporal["Filenames"] = btkTools.buildTrials(self.m_data_path,self.m_files_stp)


        # ----kinematic trials---
        if kinematicFlag:
            c3dManager.kinematic["Acqs"],c3dManager.kinematic["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_kinematic)

        #---kinetic Trials--- ( check if kinetic events)
        if kineticFlag:
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files_kinetic)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_emg)




class C3dManagerFilter(object):
    """

    """

    def __init__(self,procedure):

        self.m_procedure = procedure
        self.m_spatioTempFlag = True
        self.m_kinematicFlag = True
        self.m_kineticFlag = True
        self.m_emgFlag = True

    def enableSpatioTemporal(self, boolean):
        self.m_spatioTempFlag = boolean

    def enableKinematic(self, boolean):
        self.m_kinematicFlag = boolean

    def enableKinetic(self, boolean):
        self.m_kineticFlag = boolean

    def enableEmg(self, boolean):
        self.m_emgFlag = boolean



    def generate(self):

        c3dManager = C3dManager()


        self.m_procedure.generate(c3dManager,self.m_spatioTempFlag, self.m_kinematicFlag, self.m_kineticFlag, self.m_emgFlag)

        return c3dManager
