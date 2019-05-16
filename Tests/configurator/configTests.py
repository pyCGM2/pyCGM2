# -*- coding: utf-8 -*-
import ipdb
from pyCGM2.Utils import files
import pyCGM2
from pyCGM2.Configurator import ModelManager
import logging
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
import numpy as np
from pyCGM2.Eclipse import vskTools

pyCGM2_GLOBAL_SETTINGS_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\globalSettings\\"
pyCGM2_USER_SETTINGS_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\userSettings\\"

class tests():


    @classmethod
    def testCGM1(cls):


        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\Tests\\configurator\\localsettings\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"altered_CGM1.userSettings")

        manager = Manager.CGM1ConfigManager(userSettings)
        manager.contruct()
        manager.getFinalSettings()

        #files.prettyDictPrint(finalSettings)




    @classmethod
    def testCGM1_localSettings(cls):


        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\Tests\\configurator\\localsettings\\"

        internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"altered_CGM1.userSettings")

        manager = Manager.CGM1ConfigManager(userSettings,localInternalSettings=internalSettings)
        manager.contruct()
        finalSettings = manager.getFinalSettings()


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
        manager.contruct()
        finalSettings = manager.getFinalSettings()
        #files.prettyDictPrint(finalSettings)

        np.testing.assert_equal( finalSettings["Global"]["Marker diameter"],28)
        np.testing.assert_equal( finalSettings["Global"]["Point suffix"],"TEST")
        np.testing.assert_equal( finalSettings["Calibration"]["Left flat foot"],0)
        np.testing.assert_equal( finalSettings["Calibration"]["Right flat foot"],0)
        np.testing.assert_equal( finalSettings["Translators"]["LASI"],"VERIF")

class CGM1tests():


    @classmethod
    def userSettingsOnly(cls):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Configurator\\CGM1\\onlyUserSettings\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"CGM1.userSettings")

        manager = ModelManager.CGM1ConfigManager(userSettings)
        manager.contruct()
        manager.getFinalSettings()


    @classmethod
    def mpFromVsk(cls):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Configurator\\CGM1\\vskIn\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"CGM1.userSettings")

        vsk = vskTools.Vsk(str(DATA_PATH +  "CGM1.vsk"))

        manager = ModelManager.CGM1ConfigManager(userSettings,vsk=vsk)
        manager.contruct()
        manager.getFinalSettings()



    @classmethod
    def customTranslators(cls):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Configurator\\CGM1\\customTranslators\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"CGM1.userSettings")
        translators = files.openFile(DATA_PATH,"CGM1.translators")

        manager = ModelManager.CGM1ConfigManager(userSettings,localTranslators=translators)
        manager.contruct()
        manager.getFinalSettings()


    @classmethod
    def customAdvancedSettings(cls):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Configurator\\CGM1\\customAdvancedSettings\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"CGM1.userSettings")
        customAdvancedSettings = files.openFile(DATA_PATH,"CGM1-pyCGM2.settings")

        manager = ModelManager.CGM1ConfigManager(userSettings,localInternalSettings=customAdvancedSettings)
        manager.contruct()
        manager.getFinalSettings()


    @classmethod
    def allCustom(cls):


        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Configurator\\CGM1\\allCustom\\"

        #internalSettings = files.openFile(pyCGM2_GLOBAL_SETTINGS_PATH,"CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH,"CGM1.userSettings")
        translators = files.openFile(DATA_PATH,"CGM1.translators")
        customAdvancedSettings = files.openFile(DATA_PATH,"CGM1-pyCGM2.settings")
        vsk = vskTools.Vsk(str(DATA_PATH +  "CGM1.vsk"))


        manager = ModelManager.CGM1ConfigManager(userSettings,localInternalSettings=customAdvancedSettings,localTranslators=translators,vsk=vsk)
        manager.contruct()
        manager.getFinalSettings()




if __name__ == "__main__":
    #tests.testCGM1_localSettings()
    #tests.testCGM1_localSettingsAndTranslators()

    CGM1tests.userSettingsOnly()
    CGM1tests.mpFromVsk()
    CGM1tests.customTranslators()
    CGM1tests.customAdvancedSettings()
    CGM1tests.allCustom()
