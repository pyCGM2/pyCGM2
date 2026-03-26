# coding: utf-8
import os
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2;
LOGGER = pyCGM2.LOGGER
LOGGER.setLevel("info")
LOGGER.set_file_handler("pyCGM2-Mek.log")
from pyCGM2.Tools import uiTools
from pyCGM2.Nexus import nexus
from pyCGM2.Utils import files


def main(args=None):

    if  args is None:
        parser = argparse.ArgumentParser(description='Initialize flow report')
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)       
        args = parser.parse_args()

    data_path = args.data_path
    
    if data_path is None:
        nexusCon = nexus.NexusConnection()
        if nexusCon.isConnected():
            try:
                data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS)
            except Exception as e:
                LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
                data_path = uiTools.uiGetDir( title="Select the Session Folder (the one containing the .Session.enf file)",
                            start_dir=os.environ["USERPROFILE"]+"\\Documents")
        else:
            data_path = uiTools.uiGetDir()     

    files.createDir(data_path+"Videos")
    files.createDir(data_path+"Exams")
    files.createDir(data_path+"Images")
    files.createDir(data_path+"Doc")

    LOGGER.logger.info(f"Flow initialization completed for data path: {data_path}") 







if __name__ == "__main__":

    main(args=None)
