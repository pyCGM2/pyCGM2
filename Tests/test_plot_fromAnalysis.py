# coding: utf-8
#pytest -s --mpl --disable-pytest-warnings  test_plot_fromAnalysis.py::Test__highLevel_newNormativeData::test_gaitPanel_descriptiveKinematics

from __future__ import unicode_literals
import pytest

import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Lib import analysis, plot

from pyCGM2.Report import plot as reportPlot
from pyCGM2.Report import plotFilters,plotViewers,ComparisonPlotViewers
from pyCGM2.Report import normativeDatasets


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
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hånnibøl Lecter\\"
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

    emgFilenames = modelledFilenames


    analysisInstance = analysis.makeCGMGaitAnalysis(DATA_PATH,modelledFilenames,emgFilenames,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)


    return DATA_PATH,modelledFilenames,analysisInstance


def dataTest3():
    DATA_PATH1 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames1 = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]

    emgFilenames1 = modelledFilenames1

    analysisInstance1 = analysis.makeCGMGaitAnalysis(DATA_PATH1,modelledFilenames1,emgFilenames1,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)

    DATA_PATH2 = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 2 - CGM23\\"
    modelledFilenames2 = ["20200729-SC-PONC-S-NNNN dyn 04.c3d",
                        "20200729-SC-PONC-S-NNNN dyn 06.c3d"]

    emgFilenames2 = modelledFilenames2

    analysisInstance2 = analysis.makeCGMGaitAnalysis(DATA_PATH2,modelledFilenames2,emgFilenames2,emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None)


    return DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2



class Test_lowLevel:

    @pytest.mark.mpl_image_compare
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

    @pytest.mark.mpl_image_compare
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

    @pytest.mark.mpl_image_compare
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


class Test_highLevel:

    @pytest.mark.mpl_image_compare
    def test_highLevel_plot_spatioTemporal(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        fig = plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
                exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_plot_MAP(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_MAP(DATA_PATH,analysisInstance,normativeDataset,
            exportPdf=False,
            outputName=None,
            pointLabelSuffix=None,show=False,title=None)

        if SHOW: plt.show()
        return fig





    @pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_descriptiveKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)
        if SHOW: plt.show()
        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_consistencyKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_descriptiveKinetics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_gaitPanel_consistencyKinetics(self):
        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

        fig = plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig


    @pytest.mark.mpl_image_compare
    def test_highLevel_plotDescriptiveEnvelopEMGpanel(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        # analysis.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)

        fig = plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,analysisInstance,
                emgChannels, muscles,contexts, normalActivityEmgs,
                normalized=False,
                type="Gait",exportPdf=False,outputName=None,show=False,
                title=None)

        if SHOW: plt.show()
        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_plotDescriptiveEnvelopEMGpanel(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()

        # analysis.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)


        fig = plot.plotConsistencyEnvelopEMGpanel(DATA_PATH,analysisInstance,
                emgChannels, muscles,contexts, normalActivityEmgs,
                normalized=False,
                type="Gait",exportPdf=False,outputName=None,show=False,
                title=None)

        if SHOW: plt.show()
        return fig




    @pytest.mark.mpl_image_compare
    def test_highLevel_compareEmgEnvelops(self):
        DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2 = dataTest3()

        # analysis.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)
        analysis.normalizedEMG(analysisInstance1, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=None)
        analysis.normalizedEMG(analysisInstance2, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=analysisInstance1)

        fig = plot.compareEmgEnvelops([analysisInstance1,analysisInstance2], ["Session1", "Session2"],
            emgChannels, muscles, contexts, normalActivityEmgs,
            normalized=True,
            plotType="Descriptive",show=False,title=None,type="Gait")

        if SHOW: plt.show()
        return fig

    @pytest.mark.mpl_image_compare
    def test_highLevel_compareSelectedEmgEvelops(self):
        DATA_PATH1,modelledFilenames1,analysisInstance1,DATA_PATH2,modelledFilenames2,analysisInstance2 = dataTest3()


        # analysis.processEMG(DATA_PATH, modelledFilenames, emgChannels,
        #     highPassFrequencies=[20,200],envelopFrequency=6.0,
        #     fileSuffix=None,outDataPath=None)
        analysis.normalizedEMG(analysisInstance1, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=None)
        analysis.normalizedEMG(analysisInstance2, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=analysisInstance1)


        fig = plot.compareSelectedEmgEvelops([analysisInstance1,analysisInstance2], ["Session1", "Session2"],
                ["Voltage.EMG1","Voltage.EMG1"],["Left","Left"],normalized=True,
                plotType="Descriptive",type="Gait",show=False,title=None)

        if SHOW: plt.show()
        return fig


class Test_highLevel_customNormative:


    @pytest.mark.mpl_image_compare
    def test__highLevel_customNormative_plot_MAP(self):
        DATA_PATH,modelledFilenames,analysisInstance = dataTest2()
        normativeDataset = normativeDatasets.NormativeData("CGM23","Spont")

        fig = plot.plot_MAP(DATA_PATH,analysisInstance,normativeDataset,
            exportPdf=False,
            outputName=None,
            pointLabelSuffix=None,show=False,title=None)

        if SHOW: plt.show()
        return fig


    @pytest.mark.mpl_image_compare
    def test_highLevel_customNormative_gaitPanel_descriptiveKinematics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("CGM23","Spont")

        fig = plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)


        if SHOW: plt.show()
        return fig


    @pytest.mark.mpl_image_compare
    def test_highLevel_customNormative_gaitPanel_descriptiveKinetics(self):

        DATA_PATH,analysisInstance = dataTest1()
        normativeDataset = normativeDatasets.NormativeData("CGM23","Spont")

        fig = plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset,
            pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=False,title=None)

        if SHOW: plt.show()
        return fig
