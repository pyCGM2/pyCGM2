# coding: utf-8

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")


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


    if args  is None:
        parser = argparse.ArgumentParser(description='save to h5')
        parser.add_argument('-s','--static', action='store_true',
                            help='identiy a static trial import')
        args = parser.parse_args()

    try:
        from viconnexusapi import ViconNexus
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        data_path, filenameLabelledNoExt = NEXUS.GetTrialName()
        NEXUS.SaveTrial(30)
        LOGGER.logger.info("data Path: " + data_path)
        LOGGER.logger.info("file: " + filenameLabelledNoExt)

        type = "Static" if args.static else "Dynamic"
        LOGGER.logger.info(f" {filenameLabelledNoExt}.c3d ==> {type}")
        
        mp = mekPipe.PipeSessionSingleFileProcedure(filenameLabelledNoExt+".c3d", type=type)
        mf = mekPipe.PipeSessionBuilderFilter(mp)
        mf.run(data_path)

        h5fileOut = f"pipe/session_storage.h5"

        result = mekPipe.run_pipe(
            environment=envPathFile,
            dependencies_folder=PIPE_BASE + "pipelines/",
            pipeline=pipelineName,
            destination= f"{data_path}{h5fileOut}",
            session= f"{data_path}pipe/_session.yaml",    
            source=data_path,
            overwrite=True,
            quiet=True,
        )

        LOGGER.logger.info(f"Pipe result: {result}")


    



if __name__ == "__main__":

    main(args=None)
