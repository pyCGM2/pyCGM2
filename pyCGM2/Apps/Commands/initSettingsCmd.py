# -*- coding: utf-8 -*-

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

import argparse
import os

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='pyCGM2 data folder initialisation')
        parser.add_argument('-m', '--model', type=str,  help='CGM version')
        parser.add_argument('-e', '--emg', action='store_true',  help='copy emg settings')

        args = parser.parse_args()

    if args.emg:
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                            os.getcwd()+"\\"+"emg.settings")
        LOGGER.logger.info("[pyCGM2] file [emg.settings] copied in your data folder")

    if args.model is not None:
        if args.model == "CGM1" or args.model == "CGM1.0":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER
                            + "CGM1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM1-pyCGM2.settings] copied in your data folder")
        if args.model == "CGM1.1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM1_1-pyCGM2.settings] copied in your data folder")
        elif args.model == "CGM2.1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_1-pyCGM2.settings] copied in your data folder")
            # os.startfile(os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
        elif args.model == "CGM2.2":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_2-pyCGM2.settings] copied in your data folder")
        elif args.model == "CGM2.3":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_3-pyCGM2.settings] copied in your data folder")

        elif args.model == "CGM2.4":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_4-pyCGM2.settings] copied in your data folder")

        elif args.model == "CGM2.5":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder")

        elif args.model == "CGM2.6":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
            LOGGER.logger.info("[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder")

        else:
            LOGGER.logger.error("[pyCGM2] model version not know (CGM1, CGM1.1 ... CGM2.5)")
            raise
        


if __name__ == '__main__':
    main(args=None) 