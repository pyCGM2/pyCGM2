# coding: utf-8
import os
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2 
LOGGER = pyCGM2.LOGGER

import sys
import pandas as pd
import numpy as np
import pyCGM2
import pyCGM2;
LOGGER = pyCGM2.LOGGER
LOGGER.setLevel("info")
LOGGER.set_file_handler("pyCGM2-Mek.log")



from pyCGM2.Utils import files



from pyCGM2.Tools import uiTools
from pyCGM2 import connection

import argparse

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='Initialize flow report')
        
        
    args = parser.parse_args()

    nexusCon = connection.NexusConnection()

                            
    if nexusCon.isConnected():
        try:
            data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS) 
        except Exception as e:
            LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
            data_path = uiTools.uiGetDir()
            data_path= data_path+"\\" 
    else:
        data_path = uiTools.uiGetDir()
        data_path= data_path+"\\"     

    files.createDir(data_path+"Videos")
    files.createDir(data_path+"Exams")
    files.createDir(data_path+"Images")
    files.createDir(data_path+"Doc")

    LOGGER.logger.info(f"Flow initialization completed for data path: {data_path}") 







if __name__ == "__main__":

    main(args=None)
