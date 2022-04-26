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



class UniqueBtkAcqSetProcedure(object):
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
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files,acqs = self.m_acqs)

        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"] =self.m_acqs
            c3dManager.emg["Filenames"] = self.m_files



class UniqueC3dSetProcedure(object):
    """the same c3d filenames is used for all computational objectives

    Args:
        data_path (str): folder path
        fileLst (list): c3d filenames
    """


    def __init__(self, data_path, fileLst):
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
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files)


class DistinctC3dSetProcedure(object):
    """Distinct c3d sets are for each computational objectives

    Args:
        data_path (str): folder path
        stp_fileLst (list): c3d filenames for the spatioTemporal computation
        kinematic_fileLst (list): c3d filenames for the kinematic computation
        kinetic_fileLst (list): c3d filenames for the kinetics computation
        emg_fileLst (list): c3d filenames for the emg computation
    """

    def __init__(self, data_path, stp_fileLst, kinematic_fileLst, kinetic_fileLst, emg_fileLst):

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
            c3dManager.kinetic["Acqs"],c3dManager.kinetic["Filenames"],C3dManager.kineticFlag =  btkTools.automaticKineticDetection(self.m_data_path,self.m_files_kinetic)


        #----emgTrials
        if emgFlag:
            c3dManager.emg["Acqs"],c3dManager.emg["Filenames"], = btkTools.buildTrials(self.m_data_path,self.m_files_emg)




class C3dManagerFilter(object):
    """
    pyCGM2 Filter used for disseminate c3d trial set(s)

    Args:
        procedure ( pyCGM2.Processing.c3dManager.(Procedure)): a procedure instance
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
