# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_connection.py::Test_eclipseConnection


import pyCGM2
from pyCGM2.connection import eclipseConnector as eclDB
from pathlib import Path

from argparse import Namespace
    

class Test_eclipseConnection:
    def test_0(self):

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
            

            print("Artifact a1 full path:", svc.get_artifact_path(a1.id))

            # 6) List sessions
            for s in svc.sessions.list_by_patient(patient.ipp):
                print(f"Session {s.session_index}:", svc.get_session_path(patient.ipp, s.session_index))

        finally:
            con.close()










