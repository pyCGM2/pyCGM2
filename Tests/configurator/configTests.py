# -*- coding: utf-8 -*-
import ipdb
from pyCGM2.Utils import files
import pyCGM2

class tests():

    @classmethod
    def yaml(cls):
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2\\SessionSettings\\globalSettings\\"
        settings1_0_yaml = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.ysettings")
        settings1_0_json = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")

        ipdb.set_trace()
        settings1_1 = files.openConfigurationFile(DATA_PATH,"CGM1_1-pyCGM2.ysettings")
        settings2_1 = files.openConfigurationFile(DATA_PATH,"CGM2_1-pyCGM2.ysettings")
        settings2_2 = files.openConfigurationFile(DATA_PATH,"CGM2_2-pyCGM2.ysettings")
        settings2_3 = files.openConfigurationFile(DATA_PATH,"CGM2_3-pyCGM2.ysettings")
        settings2_4 = files.openConfigurationFile(DATA_PATH,"CGM2_4-pyCGM2.ysettings")




if __name__ == "__main__":
    tests.yaml()
