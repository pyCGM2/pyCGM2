        
import datetime
from importlib.resources import path
import pyCGM2
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


def upsert_patientClassification_to_db(path_classification, name):

    db_path = connection.DB_TEST_PATH 

    factory = eclDB.SQLiteConnectionFactory(db_path)
    con = factory.connect()

    try:
        eclDB.SchemaManager(con).ensure_schema()
        svc = eclDB.DataIndexService(con)

        root = svc.storage_roots.upsert(name=name, root_path=path_classification)  # adapt if your root is Z:\Data etc.

    finally:
        con.close()

def batchProcessing(path_classification):


    for patientDir in files.getDirs(path_classification):

        enfPatient = eclipse.PatientEnfReader(f"{path_classification}{patientDir}\\", f"{patientDir}.Patient.enf")
        ipp = enfPatient.get("PatientID")
        

        if ipp is not None:
            LOGGER.logger.info("------------------------------------------------------------------------------------------------")

            args = Namespace(  subparser="DB" ,  DB="NewPatient",   patient_path=f"{path_classification}{patientDir}\\")
            manDBcommands.main_newPatient(args)
            LOGGER.logger.info(".................................................................................................")

            for sessionDir in files.getDirs(f"{path_classification}{patientDir}", pattern=r"Session [0-9]"):
                enfSession = eclipse.SessionEnfReader(f"{path_classification}{patientDir}\\{sessionDir}\\", f"{sessionDir}.Session.enf")
                sessionIndex = utils.getNumberFromStr(sessionDir)

                args = Namespace(  subparser="DB" ,  DB="RegisterSession",   data_path=f"{path_classification}{patientDir}\\{sessionDir}\\")
                manDBcommands.main_registerSession(args)


    
    


if __name__ == "__main__":


    path_classification = "Z:\\Donnees_Nexus\\AQM Enfants\\"
    upsert_patientClassification_to_db(path_classification, "AQM Enfants")
    batchProcessing(path_classification)


