# -*- coding: utf-8 -*-
import ipdb
from pyCGM2.Utils import files
import pyCGM2
from pyCGM2.Configurator import Manager

class tests():

    @classmethod
    def testCGM1(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\"

        internalSettings = files.openFile(DATA_PATH+"globalSettings\\","CGM1-pyCGM2.settings")

        userSettings = files.openFile(DATA_PATH+"userSettings\\","CGM1.usettings")

        manager = Manager.CGM1ConfigManager(userSettings)
        manager.dynamicTrials

        ipdb.set_trace()




if __name__ == "__main__":
    tests.testCGM1()
