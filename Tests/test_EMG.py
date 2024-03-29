# -*- coding: utf-8 -*-
# @Author: Fabien Leboeuf

# pytest -s --log-cli-level=INFO --disable-pytest-warnings  test_EMG.py::Test_EMG::test_Coactivation
import numpy as np
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Lib import analysis
from pyCGM2.Lib import emg
from pyCGM2.EMG import coactivationProcedures
from pyCGM2.EMG import emgFilters
from pyCGM2.EMG import emgManager




class Test_EMG:

    def test_emgManager(self):
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"LowLevel\\emg\\"

        manager0 = emgManager.EmgManager(DATA_PATH,None)
        manager1 = emgManager.EmgManager(DATA_PATH,"emg.settings")
        manager2 = emgManager.EmgManager(DATA_PATH,"emg_altered.settings")
        manager2 = emgManager.EmgManager(None,DATA_PATH+"emg_altered.settings")

    
    def test_MVC(self):

        DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\EMG\\Normalisation\\Mvc\\"

        emgManager = emg.loadEmg(DATA_PATH)
        EMG_LABELS = emgManager.getChannels()



        # trials
        emg.processEMG(DATA_PATH, ["trial 01.c3d","trial 02.c3d","trial 03.c3d"], emgManager.getChannels(),fileSuffix ="filtered" )
        emgprocessFiles  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["trial 01.c3d","trial 02.c3d","trial 03.c3d"]]

        emgAnalysisInstance = analysis.makeAnalysis(DATA_PATH,
                            emgprocessFiles,
                            type="unknow",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = EMG_LABELS
                            )

        # MVCTA
        emg.processEMG(DATA_PATH, ["mvc TA 01.c3d","mvc TA 01.c3d","mvc TA 01.c3d"], emgManager.getChannels(),fileSuffix ="filtered" )
        emgprocessFilesMvc  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["mvc TA 01.c3d","mvc TA 01.c3d","mvc TA 01.c3d"]]
        emgAnalysisInstanceMvcTA = analysis.makeAnalysis(DATA_PATH,
                            emgprocessFilesMvc,
                            type="unknow",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = EMG_LABELS
                            )

        # MVCTA
        emg.processEMG(DATA_PATH, ["mvc SOL 01.c3d","mvc SOL 01.c3d","mvc SOL 01.c3d"], emgManager.getChannels(),fileSuffix ="filtered" )
        emgprocessFilesMvc  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["mvc SOL 01.c3d","mvc SOL 01.c3d","mvc SOL 01.c3d"]]
        emgAnalysisInstanceMvcSOL = analysis.makeAnalysis(DATA_PATH,
                            emgprocessFilesMvc,
                            type="unknow",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = EMG_LABELS
                            )


        mvc_settings = {"Voltage.EMG7": emgAnalysisInstanceMvcTA,
                        "Voltage.EMG9": emgAnalysisInstanceMvcSOL,
                        "Voltage.EMG10": emgAnalysisInstanceMvcSOL}


        emg.normalizedEMG(DATA_PATH,emgAnalysisInstance,method="MeanMax", mvcSettings=mvc_settings,fromOtherAnalysis=emgAnalysisInstance)


        # plt.plot(emgAnalysisInstance.emgStats.data['Voltage.EMG7_Rectify_Env', 'Left']["mean"])
        # plt.show()
        # import ipdb; ipdb.set_trace()

    def test_Coactivation_unithan(self):

        DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\EMG\\Normalisation\\Mvc\\"

        emgManager = emg.loadEmg(DATA_PATH)
        EMG_LABELS = emgManager.getChannels()



        # trials
        emg.processEMG(DATA_PATH, ["trial 01.c3d","trial 02.c3d","trial 03.c3d"], emgManager.getChannels(),fileSuffix ="filtered" )
        emgprocessFiles  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["trial 01.c3d","trial 02.c3d","trial 03.c3d"]]

        emgAnalysisInstance = analysis.makeAnalysis(DATA_PATH,
                            emgprocessFiles,
                            type="unknow",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = EMG_LABELS
                            )

        emg.normalizedEMG(DATA_PATH,emgAnalysisInstance,method="MeanMax", mvcSettings=None,fromOtherAnalysis=None)


        cap = coactivationProcedures.UnithanCoActivationProcedure()
        caf = emgFilters.EmgCoActivationFilter(emgAnalysisInstance,"Left")
        caf.setEMG1("Voltage.EMG1")
        caf.setEMG2("Voltage.EMG2")
        caf.setCoactivationMethod(cap)
        caf.run()



    def test_Coactivation_falconer(self):

        DATA_PATH = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\EMG\\Normalisation\\Mvc\\"

        emgManager = emg.loadEmg(DATA_PATH)
        EMG_LABELS = emgManager.getChannels()



        # trials
        emg.processEMG(DATA_PATH, ["trial 01.c3d","trial 02.c3d","trial 03.c3d"], emgManager.getChannels(),fileSuffix ="filtered" )
        emgprocessFiles  = [it[0:it.rfind(".")]+"_filtered.c3d" for it in ["trial 01.c3d","trial 02.c3d","trial 03.c3d"]]

        emgAnalysisInstance = analysis.makeAnalysis(DATA_PATH,
                            emgprocessFiles,
                            type="unknow",
                            kinematicLabelsDict=None,
                            kineticLabelsDict=None,
                            emgChannels = EMG_LABELS
                            )

        emg.normalizedEMG(DATA_PATH,emgAnalysisInstance,method="MeanMax", mvcSettings=None,fromOtherAnalysis=None)


        cap = coactivationProcedures.FalconerCoActivationProcedure()
        caf = emgFilters.EmgCoActivationFilter(emgAnalysisInstance,"Left")
        caf.setEMG1("Voltage.EMG1")
        caf.setEMG2("Voltage.EMG2")
        caf.setCoactivationMethod(cap)
        caf.run()
