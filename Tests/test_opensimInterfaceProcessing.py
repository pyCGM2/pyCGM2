# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_opensimInterfaceProcessing.py::Test_builder
import ipdb

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Lib import analysis

from pyCGM2.Report import plot as reportPlot
import matplotlib.pyplot as plt



SHOW = False

emgChannels=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4','Voltage.EMG5',
            'Voltage.EMG6','Voltage.EMG7','Voltage.EMG8','Voltage.EMG9','Voltage.EMG10']

muscles=['RF','RF','VL','VL','HAM',
            'HAM','TI','TI','SOL','SOL']

contexts=['Left','Right','Left','Right','Left',
            'Right','Left','Right','Left','Right']

normalActivityEmgs=['RECFEM','RECFEM', None,None,None,
            None,None,None,None,None]


def dataTest0():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData//CGM1-NormalGaitData-Events//Hannibal Lecter\\"
    modelledFilenames = ["gait Trial 01.c3d", "gait Trial 02.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait")

    return DATA_PATH,analysisInstance

def dataTest1_btkScalar():
    DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

    modelledFilenames = ["gait1verif.c3d", "gait2verif.c3d"]
    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                    modelledFilenames,
                    type="Gait",
                    emgChannels=None,
                    geometryMuscleLabelsDict={"Left":["add_mag2_l[MuscleLength]", "bifemlh_l[MuscleLength]"]},
                    dynamicMuscleLabelsDict = None)
    
    fig = plt.figure()
    ax = plt.gca()
    reportPlot.gaitDescriptivePlot(ax,analysisInstance.muscleGeometryStats,
                                "add_mag2_l[MuscleLength]","Left",0,
                                color="blue",
                                title="", xlabel="", ylabel="",ylim=None,
                                customLimits=None)
    plt.show()

    ipdb.set_trace

    return DATA_PATH,analysisInstance



class Test_builder:

    # def test_0(self):

    #     DATA_PATH,analysisInstance = dataTest0()
        
    def test_1(self):

        DATA_PATH,analysisInstance = dataTest1_btkScalar()



# DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

# modelledFilenames = ["gait1.c3d", "gait2.c3d"]
# analysisInstance = analysis.makeAnalysis(DATA_PATH,
#                 modelledFilenames,
#                 type="Gait",
#                 emgChannels=None)




# class Test_opensimC3dProcessing:
#     def test_processing(self):

#         DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

#         modelledFilenames = ["gait1.c3d", "gait2.c3d"]
#         analysisInstance = analysis.makeAnalysis(DATA_PATH,
#                             modelledFilenames,
#                             type="Gait")

#         ipdb.set_trace()

        



 
        
        