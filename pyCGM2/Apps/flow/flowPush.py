# coding: utf-8
import os
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2;
LOGGER = pyCGM2.LOGGER
LOGGER.setLevel("info")

from pyCGM2.Tools import uiTools
from pyCGM2.Nexus import nexus
from pyCGM2.Utils import files

from subprocess import call


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
                            

    # run 
    userSettingsFile = userSettings+".settings" if not userSettings.endswith(".settings") else userSettings
    userSettings = files.openFile(data_path,userSettingsFile)
    userSettingsFileNoExt = userSettingsFile.replace(".settings","")


    processedPath = data_path+f"Processing_{userSettingsFileNoExt}\\"

    subject_path = files.get_parent_directory(data_path)


    ipp =  userSettings["SubjectInfo"]["Ipp"]
    sessionNumber = userSettings["VisitInfo"]["SessionNumber"]
    session_dir = f"Session {sessionNumber}"
    modelVersion =  userSettings["Global"]["ModelVersion"]

    sessionPath = data_path

    
    # push vers mandbprd


    targetPath = f"{pyCGM2.FLOW_PUSH_FOLDER_PATH}{ipp}\\Session {sessionNumber}\\"
    files.createDir(targetPath)



    call(["robocopy", sessionPath+"Doc",
            targetPath+"Doc", "/S"])
    call(["robocopy", sessionPath+"Videos",
            targetPath+"Videos", "/S"])
    call(["robocopy", sessionPath+"Images",
            targetPath+"Images", "/S"])
    call(["robocopy", sessionPath+"Exams",
            targetPath+"Exams", "/S"])

    LOGGER.logger.info(f"Flow Push completed for data path: {data_path}") 


if __name__ == "__main__":

    main(args=None)
