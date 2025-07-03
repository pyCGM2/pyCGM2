# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_moveckAPI.py

import sys
import numpy as np

import pyCGM2
import pytest
from pyCGM2.Lib import analysis

from pyCGM2.Utils import files

MOVECKPATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\moveck\\"
sys.path.append(MOVECKPATH+"Moveck_pipe-2024.1.0-win64-pipeline_install\\packages")
import moveck

SHOW = False

emgChannels=['Voltage.EMG1','Voltage.EMG2','Voltage.EMG3','Voltage.EMG4','Voltage.EMG5',
            'Voltage.EMG6','Voltage.EMG7','Voltage.EMG8','Voltage.EMG9','Voltage.EMG10']

muscles=['RF','RF','VL','VL','HAM',
            'HAM','TI','TI','SOL','SOL']


contexts=['Left','Right','Left','Right','Left',
            'Right','Left','Right','Left','Right']

normalActivityEmgs=['RECFEM','RECFEM', None,None,None,
            None,None,None,None,None]


class Test_moveck:
    def test_analysis(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "OpenSim\\processingC3dOutputs\\"

        opensimSettings = files.loadSettings(DATA_PATH,"opensim.settings")
        
        modelledFilenames = ["gait1.c3d", "gait2.c3d"]
        analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        modelledFilenames,
                        type="Gait",
                        emgChannels=emgChannels,
                        geometryMuscleLabelsDict={"Left": ["glut_med1_l[MuscleLength]" , "bifemlh_l[MuscleLength]"],
                                                  "Right" : ["glut_med1_r[MuscleLength]" , "bifemlh_r[MuscleLength]"]},
                        dynamicMuscleLabelsDict = None)


        ds2 = moveck.data_store("C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\pyCGM2\\pyCGM2\\ressources\\moveck\\analysisSample.h5")

        ds = moveck.data_store()
        root = ds.root()


        stpGrp = root.create_group("SpatioTemporalData")
        for key in analysisInstance.stpStats.keys():
            stpGrp.create_set(key[0] +"/" +key[1],
                                 np.stack(analysisInstance.stpStats[key[0],key[1]]["values"], axis=-1))

        kinematicsGrp = root.create_group("kinematics")
        for key in analysisInstance.kinematicStats.data.keys():
            import ipdb; ipdb.set_trace()
            kinematicsGrp.create_set(key[0] +"/" +key[1],
                                 np.stack(analysisInstance.kinematicStats.data[key[0],key[1]]["values"], axis=-1))

        kineticsGrp = root.create_group("kinetics")
        for key in analysisInstance.kineticStats.data.keys():
            kineticsGrp.create_set(key[0] +"/" +key[1],
                                 np.stack(analysisInstance.kineticStats.data[key[0],key[1]]["values"], axis=-1))
            
        emgGrp = root.create_group("emg")
        for key in analysisInstance.emgStats.data.keys():
            emgGrp.create_set(key[0] +"/" +key[1],
                                 np.stack(analysisInstance.emgStats.data[key[0],key[1]]["values"], axis=-1))


        muscleGeoGrp = root.create_group("MTU/geometry")
        for key in analysisInstance.muscleGeometryStats.data.keys():
            muscleGeoGrp.create_set(key[0] +"/" +key[1],
                                np.stack(analysisInstance.muscleGeometryStats.data[key[0],key[1]]["values"], axis=-1))

        muscleDynGrp = root.create_group("MTU/dynamics")
        for key in analysisInstance.muscleDynamicStats.data.keys():
            muscleDynGrp.create_set(key[0] +"/" +key[1],
                                np.stack(analysisInstance.muscleDynamicStats.data[key[0],key[1]]["values"], axis=-1))


        ds.dump("storage.h5")

        import ipdb; ipdb.set_trace()

    