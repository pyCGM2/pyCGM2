# coding: utf-8
import os
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")



from pyCGM2.Utils import files
from pyCGM2.flow import flowFilters
from pyCGM2.flow.procedures import eclipseFlowProcedure
from pyCGM2.Tools import uiTools
from pyCGM2 import connection

import argparse

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='Edit flow report from Eclipse')
        parser.add_argument('-cgm', '--cgmVersion', type=str,
                            help='CGM Version from CGM1.0 to CGM2.6')
                
    args = parser.parse_args()

    version = args.cgmVersion
    versionforFile = version.replace(".","")

    nexusCon = connection.NexusConnection()

    if nexusCon.isConnected():
        try:
            data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS) 
        except Exception as e:
            LOGGER.logger.error(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
            data_path = uiTools.uiGetDir()
            data_path= data_path+"\\" 
    else:
        data_path = uiTools.uiGetDir()
        data_path= data_path+"\\"                            


    if not os.path.isfile(data_path+"emg.settings"):
        raise FileNotFoundError(f"EMG settings file not found in data path: {data_path}. Please create an emg.settings file in the data folder before running the flow edit.")  

    emgSettings = files.openFile(data_path, "emg.settings")

    fef = flowFilters.FlowEdittingFilter(data_path,version, 
                                        procedure=eclipseFlowProcedure.EclipseFlowProcedure(),
                                        emgSettings = emgSettings
                                        )

    fef.run(f"{versionforFile}_v2.settings",displayFile=True)

    LOGGER.logger.info(f"Flow editing completed for data path: {data_path} and CGM version: {version}")
    




if __name__ == "__main__":

    main(args=None)
