        
import datetime
from tokenize import Name
import pyCGM2;
LOGGER = pyCGM2.LOGGER

from pyCGM2 import enums

from pyCGM2.connection import connector 
from pyCGM2.connection import eclipseConnector as eclDB


from pyCGM2.connection import eclipseConnector as eclDB
import argparse
from pyCGM2.Utils import files
from pyCGM2.Utils import utils
from pyCGM2.Tools import uiTools

from pyCGM2.Nexus import nexus

from pyCGM2.Nexus import eclipse
import shutil

# DB_PATH = connection.DB_TEST_PATH
DB_PATH_DISTANT = pyCGM2.ECLISPE_DB_PATH 
DB_PATH_LOCAL = pyCGM2.ECLIPSE_DB_LOCAL

DB_PATH = DB_PATH_LOCAL


# copie locale

def main_newPatient(args=None,db=None):

    shutil.copy(DB_PATH_DISTANT, DB_PATH_LOCAL)

    if  args is None:
        parser = argparse.ArgumentParser(description='Add a new patient to the database')
        parser.add_argument('-pp', '--patient_path', type=str,
                            default=None)       
        args = parser.parse_args()
    
    patient_path = args.patient_path

    if patient_path is None:
        nexusCon = nexus.NexusConnection()
        if nexusCon.isConnected():
            try:
                data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS)
                patient_path = files.get_parent_directory(data_path)
            except Exception as e:
                LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}, fallback to ui selection")
                patient_path = uiTools.uiGetDir()
        else:
            patient_path = uiTools.uiGetDir()





    try:
        enfPatientFile = eclipse.getEnfFiles(patient_path,enums.EclipseType.Patient)
    except IndexError as e:
        LOGGER.logger.warning("No patient enf file found in the data path")
        raise Exception("No patient enf file found in the data path")
    
    patientDirName = patient_path.split("\\")[-2]
    classificationName = files.get_parent_directory(patient_path).split("\\")[-2]

    enfPatient = eclipse.PatientEnfReader(patient_path, enfPatientFile)
    
    ipp = enfPatient.get("PatientID")

    if ipp is not None and ipp != "":

        if db is None:
            db = DB_PATH
        
        factory = eclDB.SQLiteConnectionFactory(db)
        con = factory.connect()

        try:
            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)

            #1) Get the storage root for the classification
            root = svc.storage_roots.get_by_name(classificationName)

            #2) Register a patient
            patientDbInstance = svc.patients.upsert(ipp=ipp, folder_name=patientDirName, storage_root_id=root.id)

            LOGGER.logger.info(f"Patient {patientDbInstance.ipp} - {patientDbInstance.folder_name} registered in the database under classification ({classificationName})")   
        except Exception as e:
            LOGGER.logger.error(f"Error while registering patient in the database: {e}") 
            raise e     
        finally:
            con.close()
            shutil.copy(DB_PATH_LOCAL,DB_PATH_DISTANT)
    else:
        LOGGER.logger.warning(f"No PatientID (ipp) found in the patient enf file, cannot register patient in the database")

def main_registerSession(args=None,db=None):

    shutil.copy(DB_PATH_DISTANT, DB_PATH_LOCAL)

    if  args is None:
        parser = argparse.ArgumentParser(description='Register a session in the database')
        parser.add_argument('-dp', '--data_path', type=str,
                            default=None)       
        args = parser.parse_args()
    
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

    patient_path = files.get_parent_directory(data_path)

    try:
        enfPatientFile = eclipse.getEnfFiles(patient_path,enums.EclipseType.Patient)
    except IndexError as e:
        LOGGER.logger.warning("No patient enf file found in the data path")
        raise Exception("No patient enf file found in the data path")
    

    classificationName = files.get_parent_directory(patient_path).split("\\")[-2]
    patientDirName = patient_path.split("\\")[-2]
    sessionDirName = data_path.split("\\")[-2]
    sessionIndex = utils.getNumberFromStr(sessionDirName)


    enfPatient = eclipse.PatientEnfReader(patient_path, enfPatientFile)
    ipp = enfPatient.get("PatientID")

    enfSession = eclipse.SessionEnfReader(data_path, f"{sessionDirName}.Session.enf")
    date_exam = datetime.datetime.strptime(enfSession.get("CREATIONDATEANDTIME"), "%Y,%m,%d,%H,%M,%S").date()


    c3dFiles = files.getFiles(data_path,".c3d")
    c3d_name_meta_details={}
    for c3dFile in c3dFiles:
        
        c3dname, metadata = eclipse.getC3d_enfTrialMetadata(data_path, c3dFile)
        c3d_name_meta_details[c3dname] = metadata
        
   #---- 

    if db is None:
        db = DB_PATH


    #------ Database registration ------    
    factory = eclDB.SQLiteConnectionFactory(db)
    con = factory.connect()

    try:
        eclDB.SchemaManager(con).ensure_schema()
        svc = eclDB.DataIndexService(con)

        #1) Get the storage root for the classification
        root = svc.storage_roots.get_by_name(classificationName)

        #2) Register a patient
        if ipp is not None and ipp != "":
            patientDbInstance = svc.patients.upsert(ipp=ipp, folder_name=patientDirName, storage_root_id=root.id)
            LOGGER.logger.info(f"Patient {patientDbInstance.ipp} - {patientDbInstance.folder_name} registered in the database under classification ({classificationName})")   

        #2) get a patient
        patientDbInstance = svc.patients.get(ipp=ipp)

        # 3)Register a session
        sessionDbInstance = svc.sessions.create_or_update(ipp=patientDbInstance.ipp, session_index=sessionIndex, folder_name=sessionDirName, session_date=date_exam.strftime("%Y-%m-%d"))

        LOGGER.logger.info(f"Session {sessionDbInstance.session_index} - {sessionDbInstance.folder_name} registered in the database for patient {patientDbInstance.ipp} - {patientDbInstance.folder_name}")
    

        # 4) create / upsert the C3D artifact (relative path inside session folder)
        if c3d_name_meta_details!={}:
            for c3dName, metadataenfTrial in c3d_name_meta_details.items():
                if metadataenfTrial is not None:
                    c3dArtifactInstance = svc.artifacts.upsert(
                        session_id=sessionDbInstance.id,
                        data_type="c3d",
                        rel_path=f"{c3dName}.c3d",
                        label=c3dName
                    )

                    # 5) persist metadata as key/value rows linked to the C3D artifact
                    for key, value in metadataenfTrial.items():
                        svc.artifact_meta.upsert(
                            artifact_id=c3dArtifactInstance.id,
                            section="TRIAL_INFO",
                            key=str(key),
                            value=None if value is None else str(value),
                            value_type="str"
                        )
                    LOGGER.logger.info(f"Artifact {c3dArtifactInstance.label} ({c3dArtifactInstance.data_type}) registered in the database for session {sessionDbInstance.folder_name} with metadata")

    except Exception as e:
        LOGGER.logger.error(f"Error while registering session in the database: {e}")    
        raise e     
    finally:
        con.close()
        shutil.copy(DB_PATH_LOCAL,DB_PATH_DISTANT)


def main_patientInfo(args=None,db=None):

    if  args is None:
        parser = argparse.ArgumentParser(description='get patient info from ipp')
        parser.add_argument('--ipp',  type=str,  default=None, required=True)       
        args = parser.parse_args()

    ipp = args.ipp

    if ipp is not None and ipp != "":

        if db is None:
            db = DB_PATH
    
        factory = eclDB.SQLiteConnectionFactory(db)
        con = factory.connect()

        try:
            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)

            patientDbInstance = svc.patients.get(ipp=ipp)
            root = svc.storage_roots.get_by_id(patientDbInstance.storage_root_id)

            # patientFoldername = root.root_path + "\\" +  patientDbInstance.folder_name
            LOGGER.logger.info(f"Patient {patientDbInstance.ipp} - {patientDbInstance.folder_name} is stored in root {root.name} at path: {root.root_path}")
    
        except Exception as e:
            LOGGER.logger.error(f"Error while registering patient in the database: {e}") 
            raise e     
        finally:
            con.close()



    

if __name__ == "__main__":
    # main_patientInfo()
    pass

    
    # main_newPatient(args=None)
    # main_registerSession(args=None)
