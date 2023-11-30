# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_plot_fromAnalysis.py::Test_lowLevel::test_lowLevel_SaggitalGagePlotViewer
#pytest -s --mpl --disable-pytest-warnings  test_plot_fromAnalysis.py::Test_highLevel::test_hightLevel_SaggitalGagePlotViewer

import pytest


import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Lib import analysis, plot
from pyCGM2.Lib import emg

from pyCGM2.Report import plot as reportPlot
from pyCGM2.Report import plotFilters
from pyCGM2.Report.Viewers import plotViewers
from pyCGM2.Report.Viewers import emgPlotViewers
from pyCGM2.Report.Viewers import customPlotViewers
from pyCGM2.Report.Viewers import  comparisonPlotViewers
from pyCGM2.Report import normativeDatasets
from pyCGM2.Utils import files

from pyCGM2.EMG import emgManager

SHOW = False

emgChannels=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4','Voltage.EMG5',
            'Voltage.EMG6','Voltage.EMG7','Voltage.EMG8','Voltage.EMG9','Voltage.EMG10']

muscles=['RF','RF','VL','VL','HAM',
            'HAM','TI','TI','SOL','SOL']

contexts=['Left','Right','Left','Right','Left',
            'Right','Left','Right','Left','Right']

normalActivityEmgs=['RECFEM','RECFEM', None,None,None,
            None,None,None,None,None]


def dataTest1():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
    modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait")
    return DATA_PATH,analysisInstance


def dataTest2():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )


    return DATA_PATH,modelledFilenames,analysisInstance


def dataTest3():
    DATA_PATH1 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames1 = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    analysisInstance1 = analysis.makeAnalysis(DATA_PATH1,
                        modelledFilenames1,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )

    DATA_PATH2 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 2 - CGM23\\"
    modelledFilenames2 = ["20200729-SC-PONC-S-NNNN dyn 04.c3d",
                        "20200729-SC-PONC-S-NNNN dyn 06.c3d"]

    analysisInstance2 = analysis.makeAnalysis(DATA_PATH2,
                        modelledFilenames2,
                        type="Gait",
                        emgChannels = emgChannels,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )


    return DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2



class Test_lowLevel:

    #@pytest.mark.mpl_image_compare
    def test_gaitDescriptivePlot(self):

        DATA_PATH,analysisInstance = dataTest1()

        fig = plt.figure()
        ax = plt.gca()
        reportPlot.gaitDescriptivePlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_gaitConsistencyPlot(self):

        DATA_PATH,analysisInstance = dataTest1()

        fig = plt.figure()
        ax = plt.gca()
        reportPlot.gaitConsistencyPlot(ax,analysisInstance.kinematicStats,
                                "LPelvisAngles","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_lowLevel_NormalizedKinematicsPlotViewer(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")
        # viewer
        kv =plotViewers.NormalizedKinematicsPlotViewer(analysisInstance)
        kv.setConcretePlotFunction(reportPlot.gaitDescriptivePlot)
        kv.setNormativeDataset(normativeDataset)

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        fig = pf.plot()

        if SHOW: plt.show()
        return fig

    def test_lowLevel_SaggitalGagePlotViewer(self):

            DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
            normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")


            emg_manager = emgManager.EmgManager(None,emgSettings=None)
            #emg_manager = emgManager.EmgManager(pyCGM2.MAIN_PYCGM2_TESTS_PATH,emgSettings="emg-noRF.settings")


            # viewer
            kv =customPlotViewers.SaggitalGagePlotViewer(analysisInstance,emg_manager,emgType="Raw")
            kv.setNormativeDataset(normativeDataset)

            # filter
            pf = plotFilters.PlottingFilter()
            pf.setViewer(kv)
            fig = pf.plot()

            plt.show()

class Test_highLevel:

    #@pytest.mark.mpl_image_compare
    def test_highLevel_plot_spatioTemporal(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        fig = plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
                exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_plot_MAP(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_MAP(DATA_PATH,analysisInstance,normativeDataset,
            exportPdf=False,
            outputName=None,
            pointLabelSuffix=None,show=False,title=None)

        if SHOW: plt.show()
        return fig





    #@pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_descriptiveKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)
        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_consistencyKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_descriptiveKinetics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_consistencyKinetics(self):
        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig


    #@pytest.mark.mpl_image_compare
    def test_highLevel_plotDescriptiveEnvelopEMGpanel(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        emgManager = emg.loadEmg(DATA_PATH)
        # emgchannels = emgManager.getChannels()

        # emgSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"emg.settings")
        # emg.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)

        fig = plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,analysisInstance,
                normalized=False,
                type="Gait",exportPdf=False,outputName=None,show=False,
                title=None)

        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_plotConsistencyEnvelopEMGpanel(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        emgManager = emg.loadEmg(DATA_PATH)
        # emgchannels = emgManager.getChannels()
        # emg.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)

        fig = plot.plotConsistencyEnvelopEMGpanel(DATA_PATH,analysisInstance,
                normalized=False,
                type="Gait",exportPdf=False,outputName=None,show=False,
                title=None)

        if SHOW: plt.show()
        return fig




    #@pytest.mark.mpl_image_compare
    def test_highLevel_compareEmgEnvelops(self):
        DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2 = dataTest3()

        # emgManager = emg.loadEmg(DATA_PATH1)
        # emgchannels = emgManager.getChannels()
        # emg.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)
        emg.normalizedEMG(DATA_PATH1, analysisInstance1,  method="MeanMax", fromOtherAnalysis=None)
        emg.normalizedEMG(DATA_PATH2, analysisInstance2, method="MeanMax", fromOtherAnalysis=analysisInstance1)

        fig = plot.compareEmgEnvelops(DATA_PATH1,[analysisInstance1,analysisInstance2], ["Session1", "Session2"],
            normalized=True,
            plotType="Descriptive",show=False,title=None,type="Gait")

        if SHOW: plt.show()
        return fig

    #@pytest.mark.mpl_image_compare
    def test_highLevel_compareSelectedEmgEvelops(self):
        DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2 = dataTest3()


        # emg.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)
        emg.normalizedEMG(DATA_PATH1, analysisInstance1,  method="MeanMax", fromOtherAnalysis=None)
        emg.normalizedEMG(DATA_PATH2, analysisInstance2, method="MeanMax", fromOtherAnalysis=analysisInstance1)


        fig = plot.compareSelectedEmgEvelops(DATA_PATH1,[analysisInstance1,analysisInstance2], ["Session1", "Session2"],
                ["Voltage.EMG1","Voltage.EMG1"],["Left","Left"],normalized=True,
                plotType="Descriptive",type="Gait",show=False,title=None)

        if SHOW: plt.show()
        return fig


    def test_hightLevel_SaggitalGagePlotViewer(self):

            DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
            normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

            plot.plotSaggitalGagePanel(DATA_PATH,analysisInstance,normativeDataset,emgType="Raw")



class Test_highLevel_customNormative:


    #@pytest.mark.mpl_image_compare
    def test__highLevel_customNormative_plot_MAP(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
        normativeDataset = normativeDatasets.NormativeData("CGM23-msm","Spont")

        fig = plot.plot_MAP(DATA_PATH,analysisInstance,normativeDataset,
            exportPdf=False,
            outputName=None,
            pointLabelSuffix=None,show=False,title=None)

        if SHOW: plt.show()
        return fig


    #@pytest.mark.mpl_image_compare
    def test_highLevel_customNormative_gaitPanel_descriptiveKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("CGM23-msm","Spont")

        fig = plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)


        if SHOW: plt.show()
        return fig


    #@pytest.mark.mpl_image_compare
    def test_highLevel_customNormative_gaitPanel_descriptiveKinetics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("CGM23-msm","Spont")

        fig = plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig
