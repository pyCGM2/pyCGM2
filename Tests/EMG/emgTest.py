# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Signal import signal_processing

from pyCGM2.EMG import emgFilters

from pyCGM2.Processing import cycle,analysis,c3dManager
from pyCGM2 import enums

class test_emg():

    @classmethod
    def basicProcessing(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['EMG1','EMG2']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()





    @classmethod
    def EnveloppeProcessing(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['EMG1','EMG2']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,EMG_LABELS)
        envf.setCutoffFrequency(180.0)
        envf.run()
        btkTools.smartWriter(acq,"test.c3d")

        values = acq.GetAnalog("EMG1_Rectify").GetValues()

    @classmethod
    def psd(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['EMG1','EMG2']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,EMG_LABELS)
        envf.setCutoffFrequency(180.0)
        envf.run()
        btkTools.smartWriter(acq,"test.c3d")

        values = acq.GetAnalog("EMG1_Rectify").GetValues()


        fpcntile, mpf, fmax, Ptotal, f, P = signal_processing.psd(values[:,0],1000,show=True)

    @classmethod
    def cycleAnalysis(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['EMG1','EMG2']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,EMG_LABELS)
        envf.setCutoffFrequency(180.0)
        envf.run()

        btkTools.smartWriter(acq,DATA_PATH+"test.c3d")

        modelledFilenames = ["test.c3d"]

        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableSpatioTemporal(False)
        cmf.enableKinematic(False)
        cmf.enableKinetic(False)
        cmf.enableEmg(True)
        trialManager = cmf.generate()


        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                               kinematicTrials = trialManager.kinematic["Trials"],
                                               kineticTrials = trialManager.kinetic["Trials"],
                                               emgTrials=trialManager.emg["Trials"])

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()


        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = ['EMG1_Rectify_Env','EMG2_Rectify_Env'],
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

    @classmethod
    def NormalizationTest(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['EMG1','EMG2']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,EMG_LABELS)
        envf.setCutoffFrequency(180.0)
        envf.run()

        btkTools.smartWriter(acq,DATA_PATH+"test.c3d")

        modelledFilenames = ["test.c3d"]

        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableSpatioTemporal(False)
        cmf.enableKinematic(False)
        cmf.enableKinetic(False)
        cmf.enableEmg(True)
        trialManager = cmf.generate()


        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                               kinematicTrials = trialManager.kinematic["Trials"],
                                               kineticTrials = trialManager.kinetic["Trials"],
                                               emgTrials=trialManager.emg["Trials"])

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()


        modelInfo=None
        subjectInfo=None
        experimentalInfo=None

        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = ['EMG1_Rectify_Env','EMG2_Rectify_Env'],
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"EMG1","Left")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()
        envnf.processC3d([DATA_PATH+"test.c3d"], fileSuffix="test")


        print analysisInstance.emgStats.data.keys()


if __name__ == "__main__":
    #plt.close("all")

    #test_emg.cycleAnalysis()
    test_emg.NormalizationTest()
