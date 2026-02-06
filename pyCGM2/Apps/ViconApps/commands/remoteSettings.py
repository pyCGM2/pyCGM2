import os
import argparse
import pyCGM2

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(prog='pyCGM2-Nexus-Device')
        args = parser.parse_args()


    NEXUS_PYTHON_CONNECTED = False
    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusTools 
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.warning("Vicon nexus not connected")

    if NEXUS_PYTHON_CONNECTED:
        data_path, calibrateFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS)    

        if args.emg:
            if not os.path.isfile(data_path+"emg.settings"):
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                                    data_path+"emg.settings")
                LOGGER.logger.info(f"[pyCGM2] file [emg.settings] copied in your data folder {data_path}")
            else:
                LOGGER.logger.warning(f"[pyCGM2] file [emg.settings] already exist in your data folder {data_path}")
            os.startfile(data_path+"emg.settings")
            

        if args.model is not None:
            if args.model == "CGM1" or args.model == "CGM1.0":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER
                                + "CGM1-pyCGM2.settings", data_path+"CGM1-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM1-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM1-pyCGM2.settings")
            if args.model == "CGM1.1":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings",
                                data_path+"CGM1_1-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM1_1-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM1_1-pyCGM2.settings")
            elif args.model == "CGM2.1":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings",
                                data_path+"CGM2_1-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_1-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_1-pyCGM2.settings")
            elif args.model == "CGM2.2":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings",
                                data_path+"CGM2_2-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_2-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_2-pyCGM2.settings")
            elif args.model == "CGM2.3":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings",
                                data_path+"CGM2_3-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_3-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_3-pyCGM2.settings")

            elif args.model == "CGM2.4":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings",
                                data_path+"CGM2_4-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_4-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_4-pyCGM2.settings")

            elif args.model == "CGM2.5":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                                data_path+"CGM2_5-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_5-pyCGM2.settings")

            elif args.model == "CGM2.6":
                files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                                data_path+"CGM2_5-pyCGM2.settings")
                LOGGER.logger.info(f"[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder {data_path}")
                os.startfile(data_path+"CGM2_5-pyCGM2.settings")

            else:
                LOGGER.logger.error("[pyCGM2] model version not know (CGM1, CGM1.1 ... CGM2.5)")
                raise Exception("[pyCGM2] model version not know (CGM1, CGM1.1 ... CGM2.5)")