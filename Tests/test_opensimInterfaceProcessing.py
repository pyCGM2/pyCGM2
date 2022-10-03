# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterfaceProcessing.py::Test_opensimModelOuputprocessing_fromNexus
import ipdb
import os

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot


import matplotlib.pyplot as plt
from pyCGM2.Report import plot as reportPlot
from pyCGM2.Report import plotFilters
from pyCGM2.Report.Viewers import musclePlotViewers
from pyCGM2.Utils import files



SHOW = False

emgChannels=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4','Voltage.EMG5',
            'Voltage.EMG6','Voltage.EMG7','Voltage.EMG8','Voltage.EMG9','Voltage.EMG10']

muscles=['RF','RF','VL','VL','HAM',
            'HAM','TI','TI','SOL','SOL']

contexts=['Left','Right','Left','Right','Left',
            'Right','Left','Right','Left','Right']

normalActivityEmgs=['RECFEM','RECFEM', None,None,None,
            None,None,None,None,None]


   


class Test_opensimModelOuputprocessing_fromNexus:

    def test_specificMuscleLabels_lowLevelViewer(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        opensimSettings = files.loadSettings(DATA_PATH,"opensim.settings")
        
        modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels=None,
                        geometryMuscleLabelsDict={"Left": ["glut_med1_l[MuscleLength]" , "bifemlh_l[MuscleLength]"],
                                                  "Right" : ["glut_med1_r[MuscleLength]" , "bifemlh_r[MuscleLength]"]},
                        dynamicMuscleLabelsDict = None)
        


        # viewer
        kv =musclePlotViewers.MuscleNormalizedPlotPanelViewer(analysisInstance)
        kv.setConcretePlotFunction(reportPlot.gaitDescriptivePlot)
        kv.setMuscles(["glut_med1","bifemlh"])
        kv.setMuscleOutputType("MuscleLength")


        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        fig = pf.plot()
        plt.show()

        analysis.exportAnalysis(analysisInstance,DATA_PATH,"analysisExported.xlsx")

    def test_specificMuscleLabels_highLevelViewer(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        opensimSettings = files.loadSettings(DATA_PATH,"opensim.settings")
        
        modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels=None,
                        geometryMuscleLabelsDict={"Left": ["glut_med1_l[MuscleLength]" , "bifemlh_l[MuscleLength]"],
                                                  "Right" : ["glut_med1_r[MuscleLength]" , "bifemlh_r[MuscleLength]"]},
                        dynamicMuscleLabelsDict = None)
        
        # high-level function
        figs,filenames = plot.plot_DescriptiveMuscleLength(DATA_PATH,analysisInstance,None,exportPdf=True)

    

    def test_AllMuscleLabels_highLevelViewer(self):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        opensimSettings = files.loadSettings(DATA_PATH,"opensim.settings")

        muscleDict= {"Left": [it +"_l[MuscleLength]" for it in opensimSettings["Muscles"]],
                    "Right" : [it +"_r[MuscleLength]" for it in opensimSettings["Muscles"]]}

        
        modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels=None,
                        geometryMuscleLabelsDict=muscleDict,
                        dynamicMuscleLabelsDict = None)
        


        # gigh-level function
        figs,filenames = plot.plot_DescriptiveMuscleLength(DATA_PATH,analysisInstance,None,exportPdf=True)



