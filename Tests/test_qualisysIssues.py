# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_anomalies.py::Test_markerAnomalies::test_anomalies


import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2


from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters

from pyCGM2.Lib.CGM import cgm2_3, cgm2_5


class Test_qualisysIssues:
    def test_3digitalBertecFp(self):
        
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "Issues\\qualisys\\issue_digitalBertec_3FP\\Bertec data\\"

        staticFilename = "Static LB - CGM2 2-fromQTM.c3d"
        reconstructFilenameLabelled= "Gait LB-CGM21-fromVincent.c3d"
        # reconstructFilenameLabelled= "Gait LB - CGM2 2-fromQTM.c3d"

        markerDiameter=14
        required_mp={
        'Bodymass'   : 49.0,
        'LeftLegLength' : 905.0,
        'RightLegLength' : 905.0 ,
        'LeftKneeWidth' : 107.0,
        'RightKneeWidth' : 107.0,
        'LeftAnkleWidth' : 82.0,
        'RightAnkleWidth' : 82.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset' : 0,
        'RightShoulderOffset' : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0,
        'LeftHandThickness' : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0,
        'RightHandThickness' : 0
        }
        optional_mp = {
            'LeftTibialTorsion' : 0,
            'LeftThighRotation' : 0,
            'LeftShankRotation' : 0,
            'RightTibialTorsion' : 0,
            'RightThighRotation' : 0,
            'RightShankRotation' : 0
            }

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_5-pyCGM2.settings")

        model,finalAcqStatic,error = cgm2_5.calibrate(DATA_PATH,
            staticFilename,
            settings["Translators"],
            settings["Fitting"]["Weight"],
            required_mp,
            optional_mp,
            True,
            True,
            True,
            True,
            14,
            settings["Calibration"]["HJC"],
            None,
            displayCoordinateSystem=True,
            noKinematicsCalculation=False)


        acqGait,error = cgm2_5.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            settings["Translators"],
            settings,
            True,14.0,
            "acc2",
            "LRX",
            momentProjection =  enums.MomentProjection.JCS)


        outFilename = reconstructFilenameLabelled[:-4]+"_checked.c3d"
        btkTools.smartWriter(acqGait, str(DATA_PATH + outFilename))


