# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_connection.py::Test_eclipseConnection


import pyCGM2
from pyCGM2 import connection
from pyCGM2.connection import eclipseConnector as eclDB
from pathlib import Path

from argparse import Namespace
    
DB_PATH = connection.DB_TEST_PATH


class Test_request:
    def test_0(self):
        db_path = "C:\\Users\\fleboeuf\\Documents\DATA\\pyCGM2-Data-Tests\\NantesSamples\\eclipseDB_test.db "

        factory = eclDB.SQLiteConnectionFactory(db_path)
        con = factory.connect()

        try:
            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)
 
            root = svc.storage_roots.upsert(name="AQM Adultes", root_path="C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\NantesSamples\\AQM Adultes")  # adapt if your root is Z:\Data etc.

        finally:
            con.close()


class Test_eclipseConnection:
    def test_usageExample(self):

        db_path = "C:\\Users\\fleboeuf\\Documents\DATA\\pyCGM2-Data-Tests\\NantesSamples\\eclipseDB_test.db "

        factory = eclDB.SQLiteConnectionFactory(db_path)
        con = factory.connect()

        try:

            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)

            # 1) Define your Z: root (or any root)
            root = svc.storage_roots.upsert(name="AQM Adultes", root_path="C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\NantesSamples\\AQM Adultes")  # adapt if your root is Z:\Data etc.

            # 2) Register a patient
            # If folder name == ipp, keep it identical. Otherwise store the actual folder name.

            patient = svc.patients.upsert(ipp="666", folder_name="ABARNOU Martin", storage_root_id=root.id)

            # 3) Register sessions
            s1 = svc.sessions.create_or_update(ipp=patient.ipp, session_index=1, folder_name="Session 1", session_date="2025-02-03")
            s2 = svc.sessions.create_or_update(ipp=patient.ipp, session_index=2, folder_name="Session 2")

            # 4) Compute paths (deterministic)
            print("Session 1 path:", svc.get_session_path(patient.ipp, 1))
            print("Session 2 path:", svc.get_session_path(patient.ipp, 2))

            # 5) (Optional) Register artifacts inside a session folder (relative paths)
            a1 = svc.artifacts.upsert(session_id=s1.id, data_type="c3d", rel_path="trial01.c3d", label="Trial 01")
            
            # 6) (Optional) Register metadata artifacts inside
            svc.artifact_meta.upsert(
                            artifact_id=a1.id,
                            section="TRIAL_INFO",
                            key=str("ConditionID"),
                            value="Condition1",
                            value_type="str"
                        )


            print("Artifact a1 full path:", svc.get_artifact_path(a1.id))

 
        finally:
            con.close()

    def test_retreive(self):

        factory = eclDB.SQLiteConnectionFactory(DB_PATH)
        con = factory.connect()
        try:

            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)
            ipp="026886551"
            for s in svc.sessions.list_by_patient(ipp):
                print(f"Session {s.session_index}:", svc.get_session_path(ipp, s.session_index))
        finally:
            con.close()

    def test_retrieveIPP(self):
        factory = eclDB.SQLiteConnectionFactory(DB_PATH)
        con = factory.connect()
        ipp="030462311"
        try:

            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)

            patient = svc.patients.get(ipp=ipp)
            root = svc.storage_roots.get_by_id(patient.storage_root_id)

            patientFoldername = root.root_path + "\\" +  patient.folder_name

            print(f"Patient {patient.ipp} - {patient.folder_name} is stored in root {root.name} at path: {patientFoldername}")


        finally:
            con.close()

    def test_retreiveC3d_fromIppSession(self):

        factory = eclDB.SQLiteConnectionFactory(DB_PATH)
        con = factory.connect()
        try:

            eclDB.SchemaManager(con).ensure_schema()
            svc = eclDB.DataIndexService(con)

            items = svc.list_c3d_with_metadata(ipp="026886551", session_index=1)
            for item in items:
                print("C3D:", item.full_path)
                print("Type:", item.metadata.get(("TRIAL_INFO", "TrialType")))
                print("Condition:", item.metadata.get(("TRIAL_INFO", "ConditionID")))
        finally:
            con.close()











