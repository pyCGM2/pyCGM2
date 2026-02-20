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

def checkprocessingFolder(path_classification):


    for patientDir in files.getDirs(path_classification):

        patientPath = f"{path_classification}{patientDir}\\"

        enfPatient = eclipse.PatientEnfReader(f"{path_classification}{patientDir}\\", f"{patientDir}.Patient.enf")
        ipp = enfPatient.get("PatientID")
        

        if ipp is not None:
            LOGGER.logger.info("------------------------------------------------------------------------------------------------")

            for sessionDir in files.getDirs(f"{path_classification}{patientDir}", pattern=r"Session [0-9]"):

                sessionPath = f"{path_classification}{patientDir}\\{sessionDir}\\"

                enfSession = eclipse.SessionEnfReader(f"{path_classification}{patientDir}\\{sessionDir}\\", f"{sessionDir}.Session.enf")
                sessionIndex = utils.getNumberFromStr(sessionDir)

                
                # check if multiProcessing
                nprocDir = len(files.getDirs(sessionPath,contain="Processing-CGM"))
                if nprocDir!=1:
                    LOGGER.logger.info(f"{sessionPath}")
                    import ipdb; ipdb.set_trace()

            


    


if __name__ == "__main__":


    # path_classification = "Z:\\Donnees_Nexus\\AQM Enfants\\"

    path_classification = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\NantesSamples\\AQM Adultes\\"

    checkprocessingFolder(path_classification)

    flowReprocessing(path_classification)


