# -*- coding: utf-8 -*-
import logging
import ipdb
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Eclipse import vskTools,eclipse
from pyCGM2 import enums
import numpy as np

import os
from bs4 import BeautifulSoup
class EclipseTest():

    @classmethod
    def tests(cls):
        DATA_PATH ="C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Eclipse\\Lecter\session\\"

        trialFile ="PN01OP01S01SS02[CGM1].Trial.enf"
        trialreader = eclipse.TrialEnfReader(DATA_PATH,trialFile)
        import ipdb; ipdb.set_trace()

    @classmethod
    def readTrialFile(cls):
        DATA_PATH ="C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Eclipse\\Lecter\session\\"



        #sessionreader = eclipse.SessionEnfReader(DATA_PATH,sessionEnfFile)



        staticTrialFile ="PN01OP01S01STAT.Trial.enf"
        staticTrialreader = eclipse.TrialEnfReader(DATA_PATH,staticTrialFile)

        np.testing.assert_equal(staticTrialreader.isC3dExist(),True)
        np.testing.assert_equal(staticTrialreader.get("DESCRIPTION"),None)
        np.testing.assert_equal(staticTrialreader.get("Task"),"S-Static")
        np.testing.assert_equal(staticTrialreader.get("Shoes"),None)
        np.testing.assert_equal(staticTrialreader.get("ProthesisOrthosis"),None)
        np.testing.assert_equal(staticTrialreader.get("ExternalAid"),None)
        np.testing.assert_equal(staticTrialreader.get("PersonalAid"),None)
        np.testing.assert_equal(staticTrialreader.get("Processing"),"Ready")
        np.testing.assert_equal(staticTrialreader.get("TrialType"),"Static")
        np.testing.assert_equal(staticTrialreader.get("MarkerDiameter"),"14")
        np.testing.assert_equal(staticTrialreader.get("LeftFlatFoot"), True)
        np.testing.assert_equal(staticTrialreader.get("RightFlatFoot"), True)


        trialFile ="PN01OP01S01SS01.Trial.enf"
        trialreader = eclipse.TrialEnfReader(DATA_PATH,trialFile)

        np.testing.assert_equal(trialreader.isC3dExist(),True)

        np.testing.assert_equal(trialreader.get("FP1"),"Left")
        np.testing.assert_equal(trialreader.get("FP2"),"Right")
        np.testing.assert_equal(trialreader.get("NOTES"),None)
        np.testing.assert_equal(trialreader.get("DESCRIPTION"),None)
        np.testing.assert_equal(trialreader.get("FP3"),"Invalid")
        np.testing.assert_equal(trialreader.get("FP4"),"Invalid")
        np.testing.assert_equal(trialreader.get("Task"),"G-Normal gait")
        np.testing.assert_equal(trialreader.get("Shoes"),"B-Barefoot")
        np.testing.assert_equal(trialreader.get("ProthesisOrthosis"),"N-No help")
        np.testing.assert_equal(trialreader.get("ExternalAid"),"N-No help")
        np.testing.assert_equal(trialreader.get("PersonalAid"),"N-No help")
        np.testing.assert_equal(trialreader.get("Processing"),"Ready")
        np.testing.assert_equal(trialreader.get("TrialType"),"Motion")
        np.testing.assert_equal(trialreader.get("MarkerDiameter"),"14")




    @classmethod
    def getEnfFiles(cls):
        DATA_PATH ="C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Eclipse\Lecter\session\\"

        vskFile = vskTools.getVskFiles(DATA_PATH)

        sessionEnfFile =  eclipse.getEnfFiles(DATA_PATH,enums.EclipseType.Session)
        trialEnfFiles =  eclipse.getEnfFiles(DATA_PATH,enums.EclipseType.Trial)

        calib = eclipse.findCalibration(DATA_PATH)
        motion = eclipse.findMotions(DATA_PATH)
        kneeMotion = eclipse.findKneeMotions(DATA_PATH)


        calibProcessingOnly = eclipse.findCalibration(DATA_PATH,ignoreSelect=False)
        motionProcessingOnly = eclipse.findMotions(DATA_PATH,ignoreSelect=False)
        kneeMotionProcessingOnly = eclipse.findKneeMotions(DATA_PATH,ignoreSelect=False)

        motionClassified = eclipse.classifyEnfMotions(DATA_PATH,ignoreSelect=True)
        motionClassifiedProcessingOnly = eclipse.classifyEnfMotions(DATA_PATH,ignoreSelect=False)
        motionClassifiedProcessingOnly_TaskOnly = eclipse.classifyEnfMotions(DATA_PATH,ignoreSelect=False, criteria = ["Task"])


    @classmethod
    def currentMarkedNodesFileTest(cls):

        out = eclipse.getCurrentMarkedEnfs()

    @classmethod
    def writeTrialFile(cls):
        DATA_PATH ="C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Eclipse\\Lecter\session\\"



        #sessionreader = eclipse.SessionEnfReader(DATA_PATH,sessionEnfFile)



        staticTrialFile ="PN01OP01S01STAT.Trial.enf"
        staticTrialreader = eclipse.TrialEnfReader(DATA_PATH,staticTrialFile)
        staticTrialreader.set("Model","cgm1")
        staticTrialreader.save()


if __name__ == "__main__":

    EclipseTest.tests()
    #EclipseTest.readTrialFile()
    #EclipseTest.getEnfFiles()
    #EclipseTest.currentMarkedNodesFileTest()
    EclipseTest.writeTrialFile()
