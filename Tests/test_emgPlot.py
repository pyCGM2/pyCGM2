
# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_emgPlot.py::Test_emgPlotTests::test_temporalPlotSingleEmg


import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Tools import btkTools,trialTools
from pyCGM2.Signal import signal_processing

from pyCGM2.EMG import emgFilters

from pyCGM2.Processing import cycle,analysis,c3dManager
from pyCGM2 import enums


from pyCGM2.Report import plot,plotFilters,emgPlotViewers

class Test_emgPlotTests():

    def test_temporalPlotSingleEmg(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"EMG\\SampleNantes_latin1-çà\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['Voltage.EMG1']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)
        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()
        btkTools.smartWriter(acq,DATA_PATH+"testBasicPlot.c3d")


        trial =trialTools.smartTrialReader(DATA_PATH,"testBasicPlot.c3d")

        fig = plt.figure()
        ax = plt.gca()
        plot.temporalPlot(ax,trial,"Voltage.EMG1_Rectify",0,
                color="blue",
                title="test", xlabel="frame", ylabel="emg",ylim=None,legendLabel=None,
                customLimits=None)
        plot.addTemporalNormalActivationLayer(ax,trial,"RECFEM","Left")
        plt.show()



    def test_temporalPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"EMG\\SampleNantes_latin1-çà\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4']

        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()
        btkTools.smartWriter(acq,DATA_PATH+"testBasicPlot.c3d")


        trial =trialTools.smartTrialReader(DATA_PATH,"testBasicPlot.c3d")

        # # viewer
        kv = emgPlotViewers.TemporalEmgPlotViewer(trial)
        kv.setEmgs([["Voltage.EMG1","Left","RF"],["Voltage.EMG2","Right","RF"],["Voltage.EMG3","Left","vaste"],["Voltage.EMG4","Right","vaste"]])
        kv.setNormalActivationLabels(["RECFEM","RECFEM",None,"VASLAT"])
        kv. setEmgRectify(True)

        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.setExport(DATA_PATH,"check","png")
        pf.plot()

        plt.show()



    def test_envelopPlotSingleEmg(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['Voltage.EMG1','Voltage.EMG2']

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
                                                      emgLabelList = ['Voltage.EMG1_Rectify_Env','Voltage.EMG2_Rectify_Env'],
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"Voltage.EMG1","Left")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()



        fig = plt.figure()
        ax = plt.gca()
        plot.gaitDescriptivePlot(ax,analysisInstance.emgStats,
                                "Voltage.EMG1_Rectify_Env","Left",0,
                                color=None,
                                title="title", xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                                customLimits=None)

        footOff = analysisInstance.emgStats.pst['stancePhase', "Left"]["mean"]
        plot.addNormalActivationLayer(ax,"RECFEM", footOff)

        plt.show()


    def test_envelopCoactivationPlot(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['Voltage.EMG1','Voltage.EMG2']

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
                                                      emgLabelList = ['Voltage.EMG1_Rectify_Env','Voltage.EMG2_Rectify_Env'],
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"Voltage.EMG1","Left")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"Voltage.EMG2","Left")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()

        # viewer
        kv = emgPlotViewers.CoactivationEmgPlotViewer (analysisInstance)
        kv.setEmgs("Voltage.EMG1","Voltage.EMG2")
        kv.setMuscles("RF1","RF2")
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setContext("Left")

        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()



    def test_envelopGaitPlotPanel(self):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"EMG\\SampleNantes\\"
        gaitTrial = "gait.c3d"
        restTrial = "repos.c3d"

        EMG_LABELS=['Voltage.EMG1','Voltage.EMG2']

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
                                                      emgLabelList = ['Voltage.EMG1_Rectify_Env','Voltage.EMG2_Rectify_Env'],
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)

        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.setInfo(model = modelInfo)
        analysisFilter.build()

        analysisInstance = analysisFilter.analysis

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"Voltage.EMG1","Right")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()

        envnf = emgFilters.EmgNormalisationProcessingFilter(analysisInstance,"Voltage.EMG2","Right")
        envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        envnf.run()

        data = analysisInstance.emgStats.data


        # viewer
        kv = emgPlotViewers.EnvEmgGaitPlotPanelViewer (analysisInstance)
        kv.setEmgs([["Voltage.EMG1","Right","rf"],["Voltage.EMG2","Right","rf"]])
        kv.setNormalActivationLabels(["RECFEM","RECFEM"])
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)


        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()
