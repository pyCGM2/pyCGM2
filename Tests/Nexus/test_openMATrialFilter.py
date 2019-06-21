# -*- coding: utf-8 -*-
import ipdb
import logging
import numpy as np
import matplotlib.pyplot as plt
# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2 import ma
from pyCGM2.ma import io

from pyCGM2.Tools import trialTools
from pyCGM2.Nexus import nexusFilters





class Tests():



    @classmethod
    def croppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(filenameNoExt,subject)
        trial = trialConstructorFilter.build()



        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))


        # firstFrame = (root.findChild(ma.T_TimeSequence,"LTHI").startTime()+0.01)*100.0
        # firstFrame0 = (trial0.findChild(ma.T_TimeSequence,"LTHI").startTime()+0.01)*100.0
        #
        # lastFrame = root.findChild(ma.T_TimeSequence,"LTHI").duration()*100.0
        # lastFrame0 = trial0.findChild(ma.T_TimeSequence,"LTHI").duration()*100.0
        #
        # np.testing.assert_equal(firstFrame,firstFrame0)
        # np.testing.assert_equal(lastFrame,lastFrame0)

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            root.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            root.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)

        ma.io.write(root,"gait_cropped_checked.c3d")

    @classmethod
    def croppedC3d_noX2d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped_nox2d"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(filenameNoExt,subject)
        root = trialConstructorFilter.build()

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            root.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            root.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)


        ma.io.write(root,"gait_cropped - nox2d_checked.c3d")



    @classmethod
    def noCroppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(filenameNoExt,subject)
        root = trialConstructorFilter.build()

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))


        # firstFrame = (root.findChild(ma.T_TimeSequence,"LTHI").startTime()+0.01)*100.0
        # firstFrame0 = (trial0.findChild(ma.T_TimeSequence,"LTHI").startTime()+0.01)*100.0
        #
        # lastFrame = root.findChild(ma.T_TimeSequence,"LTHI").duration()*100.0
        # lastFrame0 = trial0.findChild(ma.T_TimeSequence,"LTHI").duration()*100.0
        #
        # np.testing.assert_equal(firstFrame,firstFrame0)
        # np.testing.assert_equal(lastFrame,lastFrame0)

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            root.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            root.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)


        ma.io.write(root,"gait_noCropped_checked.c3d")








    @classmethod
    def modelOutputs(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample0\\"
        filenameNoExt = "gait_cropped_ModelOutputDynamic"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(filenameNoExt,subject)
        root = trialConstructorFilter.build()


        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))



        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            root.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)

        ma.io.write(root,str("modelOutputsOpenMA_checked.c3d"))


class TestsX2d():


    @classmethod
    def noCroppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsX2d_noCroppedC3d_openma.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)



    @classmethod
    def croppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsX2d_croppedC3d_openma.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)

    @classmethod
    def interactiveCropping_fromNoCropped(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        NEXUS.SetTrialRegionOfInterest(300, 400)
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_interactiveCropping_fromNoCropped_openMA.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d")

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)

    @classmethod
    def interactiveCropping_fromCropped(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_withx2d\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        NEXUS.SetTrialRegionOfInterest(300, 400)
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_interactiveCropping_fromCropped_openMA.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d")

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)


class TestsNOX2d():


    @classmethod
    def noCroppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_noCroppedC3d_openma.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)



    @classmethod
    def croppedC3d(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_croppedC3d_openma.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, str(filenameNoExt+".c3d"))

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)

    @classmethod
    def interactiveCropping_fromNoCropped(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
        filenameNoExt = "gait_noCropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        NEXUS.SetTrialRegionOfInterest(300, 400)
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_interactiveCropping_fromNoCropped_openMA.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d")

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)

    @classmethod
    def interactiveCropping_fromCropped(cls):
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\NexusAPI\\BtkAcquisitionCreator\\sample_NOx2d\\"
        filenameNoExt = "gait_cropped"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )
        NEXUS.SetTrialRegionOfInterest(300, 400)
        subject = NEXUS.GetSubjectNames()[0]

        trialConstructorFilter = nexusFilters.NexusConstructTrialFilter(DATA_PATH,filenameNoExt,subject)
        trial = trialConstructorFilter.build()

        ma.io.write(trialConstructorFilter.m_root,"TestsNOX2d_interactiveCropping_fromCropped_openMA.c3d")

        trial0 = trialTools.smartTrialReader(DATA_PATH, "forCheckingInteractiveCropped\\gait_Cropped - 300-400.c3d")

        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"LTHI").data(),
                                            trial.findChild(ma.T_TimeSequence,"LTHI").data(),decimal=2)
        np.testing.assert_array_almost_equal(trial0.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),
                                            trial.findChild(ma.T_TimeSequence,"Voltage.EMG1").data(),decimal=2)






if __name__ == "__main__":

    # Tests.croppedC3d()
    # Tests.croppedC3d_noX2d()
    # Tests.noCroppedC3d()
    # Tests.modelOutputs()

    # TestsX2d.noCroppedC3d()
    # TestsX2d.croppedC3d()
    # TestsX2d.interactiveCropping_fromNoCropped()
    # TestsX2d.interactiveCropping_fromCropped()

    TestsNOX2d.noCroppedC3d()
    TestsNOX2d.croppedC3d()
    TestsNOX2d.interactiveCropping_fromNoCropped()
    TestsNOX2d.interactiveCropping_fromCropped()
