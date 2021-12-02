# coding: utf-8
# pytest -s --disable-pytest-warnings  test_btkTools.py::Test_Btk::test_btkWriter_paramAnalysisNew
# from __future__ import unicode_literals
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
import pyCGM2
LOGGER = pyCGM2.LOGGER
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk


class Test_Btk:
    def test_btkReader(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\Hannibal_c3d\\static.c3d"
        acq = btkTools.smartReader(filename, translators=None)

    def test_btkWriter(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\Hannibal_c3d\\static.c3d"
        acq = btkTools.smartReader(filename, translators=None)

        filenameOUT = pyCGM2.TEST_DATA_PATH_OUT + \
            "LowLevel\\IO\\Hannibal_c3d\\static.c3d"
        files.createDir(pyCGM2.TEST_DATA_PATH_OUT+"LowLevel\\IO\\Hannibal_c3d")
        btkTools.smartWriter(acq, filenameOUT)

    def test_appendPoint(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\Hannibal_c3d\\static.c3d"
        acq = btkTools.smartReader(filename, translators=None)
        values = acq.GetPoint("LASI").GetValues()
        btkTools.smartAppendPoint(
            acq, "LASI2", values, PointType=btk.btkPoint.Marker, desc="toto", residuals=None)

    def test_appendAnalog(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\Hannibal_c3d\\static.c3d"
        acq = btkTools.smartReader(filename, translators=None)
        values = acq.GetAnalog("Force.Fx1").GetValues()
        btkTools.smartAppendAnalog(acq, "Hän-emg", values, desc="Hän-emg")

    def test_functions(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\Hannibal_c3d\\gait1.c3d"
        acq = btkTools.smartReader(filename, translators=None)

        btkTools.GetMarkerNames(acq)
        btkTools.findNearestMarker(acq, 0, "LASI")
        btkTools.GetAnalogNames(acq)
        btkTools.isGap(acq, "LASI")
        btkTools.findMarkerGap(acq)
        btkTools.isPointExist(acq, "LASI")
        btkTools.isPointsExist(acq, ["LASI", "RASI"])

        btkTools.clearPoints(acq, ["LASI", "RASI"])
        btkTools.checkFirstAndLastFrame(acq, "LASI")
        btkTools.isGap_inAcq(acq, ["LASI", "RASI"])
        btkTools.findValidFrames(acq, ["LASI", "RASI"])

        btkTools.checkMultipleSubject(acq)
        btkTools.checkMarkers(acq, ["LASI", "RASI"])
        btkTools.clearEvents(acq, ["Foot Strike"])
        btkTools.modifyEventSubject(acq, "Hän")
        btkTools.modifySubject(acq, "Han")

        btkTools.getVisibleMarkersAtFrame(acq, ["LASI", "RASI"], 0)
        btkTools.isAnalogExist(acq, "emg-Hän")
        btkTools.constructEmptyMarker(acq, "zéros", desc="Hän")

        btkTools.getStartEndEvents(acq, "Left")

        btkTools.changeSubjectName(acq, "Hän")
        btkTools.smartGetMetadata(acq, "SUBJECTS", "USED")
        btkTools.smartSetMetadata(acq, "SUBJECTS", "USED", 0, "Hän")

    def test_btkReader_forcePlateType5(self):
        filename = pyCGM2.TEST_DATA_PATH + "LowLevel\\IO\\forcePlateType5\\hugGait.c3d"
        acq = btkTools.smartReader(filename, translators=None)

    def test_btkReader_ParamAnalysis(self):
        filename = pyCGM2.TEST_DATA_PATH + \
            "LowLevel\\IO\\\paramAnalysis\\data_paramFromNexusAPI.c3d"
        acq = btkTools.smartReader(filename, translators=None)
        parameters = btkTools.smartGetAllParamAnalysis(acq)
        parameter = btkTools.smartGetParamAnalysis(acq, "Vitesse", "Left","New Patient")


    def test_btkWriter_paramAnalysisNew(self):
        filename = pyCGM2.TEST_DATA_PATH + \
            "LowLevel\\IO\\\paramAnalysis\\data_paramFromNexusAPI.c3d"
        acq = btkTools.smartReader(filename, translators=None)
        btkTools.smartAppendParamAnalysis(acq, "new", "General", 3.0)
        # btkTools.smartWriter(acq, "testNew.c3d")

    def test_btkWriter_paramAnalysisAmend(self):
        filename = pyCGM2.TEST_DATA_PATH + \
            "LowLevel\\IO\\\paramAnalysis\\data_paramFromNexusAPI.c3d"
        acq = btkTools.smartReader(filename, translators=None)
        btkTools.smartAppendParamAnalysis(
            acq, "Vitesse", "Left", 3.5, subject="New Patient")

        # btkTools.smartWriter(acq, "testAmend.c3d")
