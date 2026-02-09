import os
import argparse
import pyCGM2

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

from pyCGM2.Tools import uiTools

from pyCGM2 import connection



def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(prog='pyCGM2-Remote EMG Settings")')
        args = parser.parse_args()

    nexusCon = connection.NexusConnection()

    if nexusCon.isConnected():
        try:
            data_path, calibrateFilenameLabelledNoExt = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS) 
        except Exception as e:
            LOGGER.logger.error(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
            data_path = uiTools.uiGetDir()
            data_path= data_path+"\\"   



    if not os.path.isfile(data_path+"emg.settings"):
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                            data_path+"emg.settings")
        LOGGER.logger.info(f"[pyCGM2] file [emg.settings] copied in your data folder {data_path}")
    else:
        LOGGER.logger.warning(f"[pyCGM2] file [emg.settings] already exist in your data folder {data_path}")
    
    os.startfile(data_path+"emg.settings")   