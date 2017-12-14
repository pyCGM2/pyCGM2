# -*- coding: utf-8 -*-
import os
import logging
import argparse
import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Model.CGM2.coreApps import cgmProcessing
from pyCGM2.Utils import files

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Gait Processing')
    parser.add_argument('-f','--file', type=str, help='processing file', default="processing.pyCGM2")
    parser.add_argument('--DEBUG', action='store_true', help='debug model')

    args = parser.parse_args()


    # --------------------------INPUTS ------------------------------------
    processingFile = args.file

    if args.DEBUG:
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\analysis\\gaitProcessing\\"
    else:
        DATA_PATH = os.getcwd()+"\\"


    processingSettings = files.openJson(DATA_PATH,processingFile)
    # --------------------INFOS ------------------------------
    # -----infos--------
    modelInfo = processingSettings["Model"]
    subjectInfo = processingSettings["Model"]

    experimentalInfo = processingSettings["ExperimentalContext"]

    for task in processingSettings["Processing"]["Tasks"]:
        experimentalInfo["Type"] = task["Type"]
        experimentalInfo.update(task["Conditions"])

        normativeData = task["Normative data"]

        modelledFilenames= task["Trials"]
        modelledFilenames = [str(x) for x in modelledFilenames]

        pointSuffix= task["PointSuffix"]
        pdfFilename = task["outputFilenameNoExt"]+ ".pdf"

        # --------------------------PROCESSING --------------------------------
        cgmProcessing.gaitprocessing(DATA_PATH,modelledFilenames,"CGM1.0",
             modelInfo, subjectInfo, experimentalInfo,
             normativeData,
             pointSuffix,
             pdfname = pdfFilename)
