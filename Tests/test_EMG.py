# coding: utf-8
# pytest -s --log-cli-level=INFO --disable-pytest-warnings  test_EMG.py::Test_EMG::test_MVC
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from pyCGM2.Lib import analysis
from pyCGM2.Lib import configuration
from pyCGM2.Lib import emg
from pyCGM2.Utils import files

class Test_EMG:
    def test_MVC(self):

        DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\EMG\\Normalisation\\Mvc\\"

        emgSettings = files.openFile(DATA_PATH,"emg.settings")
        EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES = configuration.getEmgConfiguration(None,emgSettings)

        # trials
        emg.processEMG(DATA_PATH, ["trial 01.c3d","trial 02.c3d","trial 03.c3d"], EMG_LABELS,fileSuffix ="filtered" )
        emgprocessFiles  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["trial 01.c3d","trial 02.c3d","trial 03.c3d"]]
        emgAnalysisInstance = analysis.makeEmgAnalysis(DATA_PATH, emgprocessFiles, EMG_LABELS, type="unknow")


        # MVCTA
        emg.processEMG(DATA_PATH, ["mvc TA 01.c3d","mvc TA 01.c3d","mvc TA 01.c3d"], EMG_LABELS,fileSuffix ="filtered" )
        emgprocessFilesMvc  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["mvc TA 01.c3d","mvc TA 01.c3d","mvc TA 01.c3d"]]
        emgAnalysisInstanceMvcTA = analysis.makeEmgAnalysis(DATA_PATH, emgprocessFilesMvc, EMG_LABELS, type="unknow")


        # MVCTA
        emg.processEMG(DATA_PATH, ["mvc SOL 01.c3d","mvc SOL 01.c3d","mvc SOL 01.c3d"], EMG_LABELS,fileSuffix ="filtered" )
        emgprocessFilesMvc  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["mvc SOL 01.c3d","mvc SOL 01.c3d","mvc SOL 01.c3d"]]
        emgAnalysisInstanceMvcSOL = analysis.makeEmgAnalysis(DATA_PATH, emgprocessFilesMvc, EMG_LABELS, type="unknow")

        mvc_settings = {"Voltage.EMG7": emgAnalysisInstanceMvcTA,
                        "Voltage.EMG9": emgAnalysisInstanceMvcSOL,
                        "Voltage.EMG10": emgAnalysisInstanceMvcSOL}


        emg.normalizedEMG(emgAnalysisInstance,EMG_LABELS,EMG_CONTEXT,method="MeanMax", mvcSettings=mvc_settings,fromOtherAnalysis=emgAnalysisInstance)


        # plt.plot(emgAnalysisInstance.emgStats.data['Voltage.EMG7_Rectify_Env', 'Left']["mean"])
        # plt.show()
        # import ipdb; ipdb.set_trace()
