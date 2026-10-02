# coding: utf-8
import os
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path



import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")



from pyCGM2.Utils import files
from pyCGM2.Tools import uiTools

from pyCGM2.flow import settingsHandler
from pyCGM2.Nexus import nexus


import argparse
from argparse import Namespace

try:
    from pyCGM2.Mek.pipe import pipe as mekPipe
except ImportError as e:
    LOGGER.logger.error(f"Error importing Mek modules: {e}. Mek functionalities will not be available.")
    raise e    


PIPE_BASE = pyCGM2.PYCGM2_SETTINGS_FOLDER+"pipe/"
envPathFile = PIPE_BASE + "config-env.yaml"
pipelineName = f"pycgm2.preparation.transform-c3d-to-device-to-h5"


def main(args=None):


    if args is None:
        parser = argparse.ArgumentParser(description='Process flow report from Eclipse')
        parser.add_argument('-u', '--userSettings', type=str,
                            help='userSettings file name, should be in the data folder',
                            required=True)
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)  
     


      

        args = parser.parse_args()
    

    userSettings = args.userSettings
    data_path = args.data_path

    h5fileOut = f"storage.h5"


    if data_path is None:
        nexusCon = nexus.NexusConnection()
        if nexusCon.isConnected():
            try:
                data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS)
                
            except Exception as e:
                LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
                data_path = uiTools.uiGetDir()
        else:
            data_path = uiTools.uiGetDir() 


    userSettingsFile = userSettings+".settings" if not userSettings.endswith(".settings") else userSettings
    userSettings = files.openFile(data_path,userSettingsFile)




       
    mp = mekPipe.PipeSessionFlowUserSettingsProcedure(userSettings)
    mf = mekPipe.PipeSessionBuilderFilter(mp)
    mf.run(data_path)
    
    

    h5fileOut = f"pipe/flow_session_storage.h5"

    result = mekPipe.run_pipe(
        environment= envPathFile,
        dependencies_folder=PIPE_BASE + "pipelines/",
        pipeline=pipelineName,
        destination= f"{data_path}{h5fileOut}",
        session= f"{data_path}pipe/pipe_flow_session.yaml",    
        source=data_path,
        overwrite=True,
        quiet=True,
    )

    LOGGER.logger.info(f"Pipe result: {result}")


if __name__ == "__main__":

    main(args=None)
