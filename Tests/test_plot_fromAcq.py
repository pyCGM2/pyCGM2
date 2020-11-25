# coding: utf-8
# pytest -s --disable-pytest-warnings  test_plot_fromAcq.py::Test_lowLevel::test_temporalPlot
# from __future__ import unicode_literals
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.EMG import emgFilters

from pyCGM2.Tools import btkTools
from pyCGM2.Lib import analysis, plot

from pyCGM2.Report import plot as reportPlot
from pyCGM2.Report import plotFilters,emgPlotViewers

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
    acq = btkTools.smartReader(DATA_PATH+modelledFilenames[0])

    return DATA_PATH, modelledFilenames,acq

def dataTest2():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\session 1 - CGM1\\"
    modelledFilenames = ["20180706_CS_PONC_S_NNNN dyn 02.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 03.c3d",
                        "20180706_CS_PONC_S_NNNN dyn 05.c3d"]
    acq = btkTools.smartReader(DATA_PATH+modelledFilenames[0])

    return DATA_PATH, modelledFilenames,acq


class Test_lowLevel:

    # @pytest.mark.mpl_image_compare
    def test_temporalPlot(self):
        DATA_PATH, modelledFilenames,acq = dataTest1()
        fig = plt.figure()
        ax = plt.gca()
        reportPlot.temporalPlot(ax,acq,"LPelvisAngles",0,color="blue",
                title="test", xlabel="frame", ylabel="angle",ylim=None,legendLabel=None,
                customLimits=None)

        if SHOW: plt.show()
        return fig


    #@pytest.mark.mpl_image_compare
    def test_lowLevel_temporalEmgPlot_4channels(self):

        DATA_PATH, modelledFilenames,acq = dataTest2()

        fig = plt.figure()


        EMG_LABELS=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4']

        bf = emgFilters.BasicEmgProcessingFilter(acq,EMG_LABELS)
        bf.setHighPassFrequencies(20.0,200.0)
        bf.run()

        # # viewer
        kv = emgPlotViewers.TemporalEmgPlotViewer(acq)
        kv.setEmgs([["Voltage.EMG1","Left","RF"],["Voltage.EMG2","Right","RF"],
                    ["Voltage.EMG3","Left","vaste"],["Voltage.EMG4","Right","vaste"]])
        kv.setNormalActivationLabels(["RECFEM","RECFEM",None,"VASLAT"])
        kv. setEmgRectify(True)

        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        fig = pf.plot()
        # plt.show()

        if SHOW: plt.show()
        return fig


class Test_highLevel:

    #@pytest.mark.mpl_image_compare
    def test_temporalKinematicPlot(self):
        DATA_PATH, modelledFilenames,acq = dataTest1()
        fig = plot.plotTemporalKinematic(DATA_PATH, modelledFilenames[0],"LowerLimb",
            pointLabelSuffix=None, exportPdf=False,outputName=None,show=False,
            title=None,       btkAcq=None)

        if SHOW: plt.show()
        return fig


    #@pytest.mark.mpl_image_compare
    def test_temporalEmgPlot(self):
        DATA_PATH, modelledFilenames,acq = dataTest2()

        analysis.processEMG(DATA_PATH, modelledFilenames, emgChannels,
            highPassFrequencies=[20,200],envelopFrequency=6.0,
            fileSuffix=None,outDataPath=None)


        figs = plot.plotTemporalEMG(DATA_PATH, modelledFilenames[0],
                emgChannels, muscles, contexts, normalActivityEmgs,
                rectify = True,
                exportPdf=False,outputName=None,show=False,title=None,
                btkAcq=None)

        if SHOW: plt.show()
        return figs[0]

plt.show()
