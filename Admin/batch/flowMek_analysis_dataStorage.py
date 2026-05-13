import datetime
from importlib.resources import path
import pyCGM2
import os
from pyCGM2.Apps.manDB import manDBcommands;
LOGGER = pyCGM2.LOGGER

from pyCGM2 import connection 
from pyCGM2.connection import eclipseConnector as eclDB
import argparse
from pyCGM2.Utils import files
from pyCGM2.Utils import utils
from pyCGM2.Tools import uiTools

from argparse import Namespace
from pyCGM2.Apps.manDB import manDBcommands

DB_PATH = connection.DB_TEST_PATH 

from pyCGM2.Nexus import eclipse


from pyCGM2.Apps.flow import flowInit
from pyCGM2.Apps.flow import flowEdit
from pyCGM2.Apps.flow import flowPrepare
from pyCGM2.Apps.flow import flowMekPopulate

import pandas as pd


def flowReprocessing(path_classification):


    for patientDir in files.getDirs(path_classification):

        patientPath = f"{path_classification}{patientDir}\\"

        enfPatient = eclipse.PatientEnfReader(f"{path_classification}{patientDir}\\", f"{patientDir}.Patient.enf")
        ipp = enfPatient.get("PatientID")
        
        h5files =  files.getFiles(f"{path_classification}{patientDir}\\",".h5")
        for it in h5files:
            os.remove(patientPath+it)


        if ipp is not None:
            LOGGER.logger.info("------------------------------------------------------------------------------------------------")

            for sessionDir in files.getDirs(f"{path_classification}{patientDir}", pattern=r"Session [0-9]"):

                sessionPath = f"{path_classification}{patientDir}\\{sessionDir}\\"

                enfSession = eclipse.SessionEnfReader(f"{path_classification}{patientDir}\\{sessionDir}\\", f"{sessionDir}.Session.enf")
                sessionIndex = utils.getNumberFromStr(sessionDir)

                folders = files.getDirs(sessionPath,contain="_v2")

                if folders != []:
                    for it in folders:
                        files.deleteDirectory(sessionPath+it)
                

                previousprocessingDir = files.getDirs(sessionPath,contain="Processing-CGM")
                CGMversionShort = previousprocessingDir[0].rsplit("-", 1)[-1]
                CGMversion = f"{CGMversionShort[:4]}.{CGMversionShort[4:]}"

                LOGGER.logger.info(f"{CGMversion}")

                

            
                args = Namespace(  subparser="FLOW" ,  DB="Edit",   
                                 data_path=f"{path_classification}{patientDir}\\{sessionDir}\\",
                                 cgmVersion=CGMversion,
                                 suffix="",
                                 display=False)
            
                flowEdit.main(args)

                args = Namespace(  subparser="FLOW" ,  DB="Prepare", 
                                 userSettings =f"{CGMversionShort}_v2.settings",  
                                 data_path=f"{path_classification}{patientDir}\\{sessionDir}\\",
                                 conditions=None)
            
                flowPrepare.main(args)



                args = Namespace(  subparser="FLOW" ,  DB="Populate", 
                                 userSettings =f"{CGMversionShort}_v2.settings",  
                                 data_path=f"{path_classification}{patientDir}\\{sessionDir}\\",
                                 analysisID=1,
                                 update = True,
                                 conditions=None)
            
                flowMekPopulate.main(args)

# def checkprocessingFolder(path_classification):

#      for patientDir in files.getDirs(path_classification):

#         patientPath = f"{path_classification}{patientDir}\\"

#         enfPatient = eclipse.PatientEnfReader(f"{path_classification}{patientDir}\\", f"{patientDir}.Patient.enf")
#         ipp = enfPatient.get("PatientID")
        

#         if ipp is not None:
#             LOGGER.logger.info("------------------------------------------------------------------------------------------------")

#             for sessionDir in files.getDirs(f"{path_classification}{patientDir}", pattern=r"Session [0-9]"):

#                 sessionPath = f"{path_classification}{patientDir}\\{sessionDir}\\"

#                 enfSession = eclipse.SessionEnfReader(f"{path_classification}{patientDir}\\{sessionDir}\\", f"{sessionDir}.Session.enf")
#                 sessionIndex = utils.getNumberFromStr(sessionDir)

#                 try:
#                     newfolder = files.getDirs(sessionPath,contain="_v2")[0]
#                 except IndexError:
#                     newfolder = None


#                 folders = files.getDirs(sessionPath, contain="Processing-CGM", pattern=r"(?!.*_v2).*")

#                 if newfolder is not None and folders != []:
#                     LOGGER.logger.info(f"{sessionPath} : Flow Processing Version 2 only")
#                 elif newfolder is None and folders != []:
#                     LOGGER.logger.info(f"{sessionPath} :Flow Processing Version 1 only")
#                     nprocDir = len(folders)
#                     if nprocDir!=1:
#                         LOGGER.logger.info(f"{sessionPath}")
#                         import ipdb; ipdb.set_trace()
#                 if newfolder is  None and folders == []:
#                     LOGGER.logger.info(f"{sessionPath} :NO Flow Processings ")

def checkprocessingFolder(path_classification) -> pd.DataFrame:

    rows = []

    for patientDir in files.getDirs(path_classification):

        patientPath = f"{path_classification}{patientDir}\\"
        enfPatient = eclipse.PatientEnfReader(patientPath, f"{patientDir}.Patient.enf")
        ipp = enfPatient.get("PatientID")

        if ipp is not None:
            for sessionDir in files.getDirs(f"{path_classification}{patientDir}", pattern=r"Session [0-9]"):

                sessionPath = f"{path_classification}{patientDir}\\{sessionDir}\\"
                sessionIndex = utils.getNumberFromStr(sessionDir)

                try:
                    newfolder = files.getDirs(sessionPath, contain="_v2")[0]
                except IndexError:
                    newfolder = None

                folders = files.getDirs(sessionPath, contain="Processing-CGM", pattern=r"(?!.*_v2).*")

                import ipdb; ipdb.set_trace()
                if newfolder is not None and folders != []:
                    status = "Flow Processing Version 2 and Version 1"
                if newfolder is not None and folders == []:
                    status = "Flow Processing Version 2 only"
                elif newfolder is None and folders != []:
                    status = "Flow Processing Version 1 only"
                    nprocDir = len(folders)
                    if nprocDir != 1:
                        status = f"Flow Processing Version 1 - multiple folders ({nprocDir})"
                elif newfolder is None and folders == []:
                    status = "NO Flow Processings"
                else:
                    status = "Unknown"

                rows.append({
                    "IPP":          ipp,
                    "Patient":      patientDir,
                    "Session":      sessionDir,
                    "SessionIndex": sessionIndex,
                    "Status":       status,
                    "Path":         sessionPath,
                })

    df = pd.DataFrame(rows, columns=["IPP", "Patient", "Session", "SessionIndex", "Status", "Path"])
    LOGGER.logger.info(f"\n{df.to_string()}")
    return df            


    


if __name__ == "__main__":


    path_classification = "Z:\\Donnees_Nexus\\AQM Enfants\\"

    # path_classification = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\NantesSamples\\AQM Adultes\\"

    df = checkprocessingFolder(path_classification)
    import ipdb; ipdb.set_trace()

    #flowReprocessing(path_classification)


