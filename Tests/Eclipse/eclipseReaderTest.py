# -*- coding: utf-8 -*-
import logging
import ipdb
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Eclipse import vskTools,eclipse
from pyCGM2 import enums

class EclipseTest():


    @classmethod
    def test(cls):
        DATA_PATH ="C:\Users\HLS501\Documents\VICON DATA\pyCGM2-Data\Eclipse\PN01\PN01OP01S01\\"

        vskFile = vskTools.getVskFiles(DATA_PATH)

        sessionEnfFile =  eclipse.getEnfFiles(DATA_PATH,enums.EclipseType.Session)
        sessionreader = eclipse.SessionEnfReader(DATA_PATH,sessionEnfFile)



        trialEnfFiles =  eclipse.getEnfFiles(DATA_PATH,enums.EclipseType.Trial)
        trialreader = eclipse.TrialEnfReader(DATA_PATH,trialEnfFiles[0])
        trialreader.isC3dExist()


        calib = eclipse.findCalibration(DATA_PATH)
        calibtrialreader = eclipse.TrialEnfReader(DATA_PATH,calib)


        motion = eclipse.findMotions(DATA_PATH)
        motionTrialreader = eclipse.TrialEnfReader(DATA_PATH,motion[0])

        import ipdb; ipdb.set_trace()


if __name__ == "__main__":


    EclipseTest.test()
