# -*- coding: utf-8 -*-
import ipdb
from pyCGM2.Utils import files
import pyCGM2
from pyCGM2.Configurator import Manager
import logging
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
import numpy as np

pyCGM2_GLOBAL_SETTINGS_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\globalSettings\\"
pyCGM2_USER_SETTINGS_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\userSettings\\"

class tests():


    @classmethod
    def testCGM1_localSettings(cls):


        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\Tests\\configurator\\localsettings\\"

        internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"altered_CGM1.userSettings")

        manager = Manager.CGM1ConfigManager(userSettings,localInternalSettings=internalSettings)
        finalSettings = manager.fullSettings()

        #files.prettyDictPrint(finalSettings)

        np.testing.assert_equal( finalSettings["Global"]["Marker diameter"],28)
        np.testing.assert_equal( finalSettings["Global"]["Point suffix"],"TEST")
        np.testing.assert_equal( finalSettings["Calibration"]["Left flat foot"],0)
        np.testing.assert_equal( finalSettings["Calibration"]["Right flat foot"],0)
        np.testing.assert_equal( finalSettings["Translators"]["LASI"],"None")



    @classmethod
    def testCGM1_localSettingsAndTranslators(cls):


        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\Tests\\configurator\\localsettings\\"

        internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        translators = files.openFile(DATA_PATH,"altered_CGM1.translators")
        userSettings = files.openFile(DATA_PATH,"altered_CGM1.userSettings")


        manager = Manager.CGM1ConfigManager(userSettings,localInternalSettings=internalSettings,localTranslators=translators)
        finalSettings = manager.fullSettings()
        #files.prettyDictPrint(finalSettings)

        np.testing.assert_equal( finalSettings["Global"]["Marker diameter"],28)
        np.testing.assert_equal( finalSettings["Global"]["Point suffix"],"TEST")
        np.testing.assert_equal( finalSettings["Calibration"]["Left flat foot"],0)
        np.testing.assert_equal( finalSettings["Calibration"]["Right flat foot"],0)
        np.testing.assert_equal( finalSettings["Translators"]["LASI"],"VERIF")




if __name__ == "__main__":
    tests.testCGM1_localSettings()
    tests.testCGM1_localSettingsAndTranslators()
