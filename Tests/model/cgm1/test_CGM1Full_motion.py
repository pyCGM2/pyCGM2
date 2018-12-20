# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp

import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2 import enums


def getViconRmatrix(frameVal, acq, originLabel, proximalLabel, lateralLabel, sequence):

        pt1 = acq.GetPoint(originLabel).GetValues()[frameVal,:]
        pt2 = acq.GetPoint(proximalLabel).GetValues()[frameVal,:]
        pt3 = acq.GetPoint(lateralLabel).GetValues()[frameVal,:]

        a1 = (pt2-pt1)
        a1 = a1/np.linalg.norm(a1)
        v = (pt3-pt1)
        v = v/np.linalg.norm(v)
        a2 = np.cross(a1,v)
        a2 = a2/np.linalg.norm(a2)
        x,y,z,R = frame.setFrameData(a1,a2,sequence)

        return R


class CGM1_motionTest():


    @classmethod
    def CGM1_UpperLimb(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\full-PiG\\"
        staticFilename = "PN01NORMSTAT.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure(bodyPart=enums.BodyPart.UpperLimb)


        markerDiameter=14
        mp={
        'LeftShoulderOffset'   : 50,
        'LeftElbowWidth' : 91,
        'LeftWristWidth' : 56 ,
        'LeftHandThickness' : 28 ,
        'RightShoulderOffset'   : 45,
        'RightElbowWidth' : 90,
        'RightWristWidth' : 55 ,
        'RightHandThickness' : 30         }
        model.addAnthropoInputParameters(mp)

         # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()
        csp = modelFilters.ModelCoordinateSystemProcedure(model)


        # --- motion ----
        gaitFilename="PN01NORMSS01.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
        modMotion.compute()

        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

        #   thorax
        R_thorax= model.getSegment("Thorax").anatomicalFrame.motion[10].getRotation()
        R_thorax_vicon = getViconRmatrix(10, acqGait, "TRXO", "TRXA", "TRXL", "XZY")
        np.testing.assert_almost_equal( R_thorax,
                                R_thorax_vicon, decimal =3)

        #   head
        R_head= model.getSegment("Head").anatomicalFrame.motion[10].getRotation()
        R_head_vicon = getViconRmatrix(10, acqGait, "HEDO", "HEDA", "HEDL", "XZY")

        np.testing.assert_almost_equal( R_head,
                                R_head_vicon, decimal =2)









if __name__ == "__main__":



    CGM1_motionTest.CGM1_UpperLimb()



    logging.info("######## PROCESS CGM1 --> Done ######")
