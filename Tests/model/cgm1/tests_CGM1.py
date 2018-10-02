# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 13:59:48 2016

@author: aaa34169


I prefer numpy.testing than unitest.
easy to debug and assert method better suits.

"""

# -*- coding: utf-8 -*-
import os
import logging
#from pyCGM2 import log
import matplotlib.pyplot as plt
import numpy as np


import ipdb

# pyCGM2
import pyCGM2
from pyCGM2 import enums
from pyCGM2.Tools from pyCGM2 import btkTools
from pyCGM2.Utils import files
from pyCGM2.Model.CGM2.coreApps import cgmUtils, cgm1, cgmProcessing
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Model import modelFilters, frame

#from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

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

def testAnatomicReferential(model,acqGait):
    # ---     tests on anatomical referential
    R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
    R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
    R_leftShankprox= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()

    R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

    R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
    R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
    R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

    #       thigh
    R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
    R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

    np.testing.assert_almost_equal( R_leftThigh,
                                    R_leftThighVicon, decimal =3)
    np.testing.assert_almost_equal( R_rightThigh,
                                    R_rightThighVicon, decimal =3)

    #       shank
    R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY")
    R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")

    np.testing.assert_almost_equal( R_leftShank,
                                    R_leftShankVicon, decimal =3)
    np.testing.assert_almost_equal( R_rightShank,
                                    R_rightShankVicon, decimal =3)

    #       foot
    R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
    R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
    np.testing.assert_almost_equal( R_leftFoot,
                                    R_leftFootVicon, decimal =3)
    np.testing.assert_almost_equal( R_rightFoot,
                                    R_rightFootVicon, decimal =3)




def offsetTesting(acqStatic,model,display = False, unitTesting=False):
    # Display
    # tibial torsion
    ltt_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LTibialTorsion").value().GetInfo().ToDouble()[0])
    rtt_vicon =np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RTibialTorsion").value().GetInfo().ToDouble()[0])


    # shank abAdd offset
    abdAdd_l = model.getViconAnkleAbAddOffset("Left")
    abdAdd_r = model.getViconAnkleAbAddOffset("Right")

    # thigh and shank Offsets
    lto = model.getViconThighOffset("Left")
    lso = model.getViconShankOffset("Left")
    rto = model.getViconThighOffset("Right")
    rso = model.getViconShankOffset("Right")

    # foot offsets
    spf_l,sro_l= model.getViconFootOffset("Left")
    spf_r,sro_r= model.getViconFootOffset("Right")

    vicon_abdAdd_l = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LAnkleAbAdd").value().GetInfo().ToDouble()[0])
    vicon_abdAdd_r = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RAnkleAbAdd").value().GetInfo().ToDouble()[0])


    lto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0])
    lso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0])

    rto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0])
    rso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0])

    vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
    vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
    vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
    vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])

    if display:
        logging.info(" LTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"]))
        logging.info(" RTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"]))
        logging.info(" LAbdAddRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_abdAdd_l,abdAdd_l))
        logging.info(" RAbdAddRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_abdAdd_r,abdAdd_r))
        logging.info(" LThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lto_vicon,lto))
        logging.info(" LShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lso_vicon,lso))
        logging.info(" RThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rto_vicon,rto))
        logging.info(" RShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rso_vicon,rso))
        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_spf_l, spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_spf_r, spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_sro_l, sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_sro_r ,sro_r))

    if unitTesting:

        # tibial torsion
        np.testing.assert_almost_equal(-ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"] , decimal = 3)
        np.testing.assert_almost_equal(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)


        # thigh and shank Offsets
        np.testing.assert_almost_equal(lto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)
        np.testing.assert_almost_equal(lso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)

        np.testing.assert_almost_equal(rto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)
        np.testing.assert_almost_equal(rso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)

        # shank abAdd offset
        np.testing.assert_almost_equal(abdAdd_l ,vicon_abdAdd_l  , decimal = 3)
        np.testing.assert_almost_equal(abdAdd_r ,vicon_abdAdd_r  , decimal = 3)

        # foot offsets
        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

def plotRelativesAngle(acqGait,suffix, label):

    pointSuffix =  "_" + suffix if suffix!="" else ""


    plt.figure()
    plt.plot( acqGait.GetPoint(label).GetValues(),"-g")
    plt.plot( acqGait.GetPoint(label+pointSuffix).GetValues(),"+r")
    plt.title(label)
    plt.show()


def testRelativesAngles(acqGait,suffix, labels = None,plotError =  False):

    pointSuffix =  "_" + suffix if suffix!="" else ""

    if labels is None:
        labels = ["RHipAngles","LHipAngles","RKneeAngles","LKneeAngles","RAnkleAngles","LAnkleAngles"]

    if not plotError:
        for label in labels:
            np.testing.assert_almost_equal( acqGait.GetPoint(label).GetValues(),
                                            acqGait.GetPoint(label+pointSuffix).GetValues(), decimal=1)
    else:
        for label in labels:
            try:
                np.testing.assert_almost_equal( acqGait.GetPoint(label).GetValues(),
                                                acqGait.GetPoint(label+pointSuffix).GetValues(), decimal =2)
            except AssertionError:
                plt.figure()
                plt.plot( acqGait.GetPoint(label).GetValues(),"-g")
                plt.plot( acqGait.GetPoint(label+pointSuffix).GetValues(),"+r")
                plt.title(label)
                plt.show()


def testJointCentres(acqGait, cgm1Label = None, pigLabel = None):

    if cgm1Label is not None and pigLabel is not None:
         np.testing.assert_almost_equal( acqGait.GetPoint(pigLabel).GetValues(),
                                            acqGait.GetPoint(cgm1Label).GetValues(), decimal =2)

    elif cgm1Label is None and pigLabel is None:
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =2)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =1)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =2)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =2)
    else:
        raise exception ("you must define both labels ")
        # -*- coding: utf-8 -*-


class CGM1_Tests():


    @classmethod
    def KadMed_TrueEquinus(cls):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\kad-med-TrueEquinus\\"
        staticFilename = "static.c3d"


        markerDiameter=14
        required_mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665.0,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 0
        }


        #import ipdb; ipdb.set_trace()
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
        translators = settings["Translators"]

        pointSuffix = "cgm1_6dof"

        model,acqStatic = cgm1.calibrate(DATA_PATH,staticFilename,translators,
                                   required_mp,optional_mp,
                                   False,False,markerDiameter,
                                   pointSuffix)


        btkTools.smartWriter(acqStatic,"CGM1-KadMed-TrueEquinus-static.c3d")


        #motion
        gaitFilename="gait trial 01.c3d"
        mfpa  = None
        momentProjection=enums.MomentProjection.Distal

        acqGait = cgm1.fitting(model,DATA_PATH, gaitFilename,
                 translators,
                 markerDiameter,
                 pointSuffix,
                 mfpa,
                 momentProjection)

        mcsp = modelFilters.ModelCoordinateSystemProcedure(model)
        mcsf = modelFilters.CoordinateSystemDisplayFilter(mcsp,model,acqGait)
        mcsf.setStatic(False)
        mcsf.display()

        #mcsp = modelFilters.ModelCoordinateSystemProcedure(model)
        #modelFilters.CoordinateSystemDisplayFilter(mcsp,model,acqGait).display()


        btkTools.smartWriter(acqGait, "CGM1-KadMed-TrueEquinus-gait.c3d")


        # testings
        #offsetTesting(acqStatic,model,display = True, unitTesting=True)
        #testJointCentres(acqGait)

        plotRelativesAngle(acqGait,pointSuffix,"LAnkleAngles")

        #testAnatomicReferential(model,acqGait)
        testRelativesAngles(acqGait,pointSuffix)


    @classmethod
    def KadMed_TrueEquinus_leftSkinMarkers(cls):
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\kad-med-TrueEquinus-leftSkinMarkers\\"
        staticFilename = "static.c3d"


        markerDiameter=14
        required_mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665.0,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 0
        }


        #import ipdb; ipdb.set_trace()
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
        translators = settings["Translators"]

        pointSuffix = "cgm1_6dof"

        model,acqStatic = cgm1.calibrate(DATA_PATH,staticFilename,translators,
                                   required_mp,optional_mp,
                                   False,False,markerDiameter,
                                   pointSuffix)

        mcsp = modelFilters.ModelCoordinateSystemProcedure(model)
        mcsf = modelFilters.CoordinateSystemDisplayFilter(mcsp,model,acqStatic)
        mcsf.setStatic(False)
        mcsf.display()
        btkTools.smartWriter(acqStatic,"CGM1-KadMed-TrueEquinus-leftSkin-static.c3d")


        #motion
        gaitFilename="gait trial 01.c3d"
        mfpa  = None
        momentProjection=enums.MomentProjection.Distal

        acqGait = cgm1.fitting(model,DATA_PATH, gaitFilename,
                 translators,
                 markerDiameter,
                 pointSuffix,
                 mfpa,
                 momentProjection)


        mcsp = modelFilters.ModelCoordinateSystemProcedure(model)
        mcsf = modelFilters.CoordinateSystemDisplayFilter(mcsp,model,acqGait)
        mcsf.setStatic(False)
        mcsf.display()

        #mcsp = modelFilters.ModelCoordinateSystemProcedure(model)
        #modelFilters.CoordinateSystemDisplayFilter(mcsp,model,acqGait).display()


        btkTools.smartWriter(acqGait, "CGM1-KadMed-TrueEquinus-leftSkin-gait.c3d")


        # testings
        offsetTesting(acqStatic,model,display = True, unitTesting=True)
        testJointCentres(acqGait)
        testRelativesAngles(acqGait,pointSuffix)

if __name__ == "__main__":

    # CGM 1
    logging.info("######## PROCESS CGM1 ######")

    #CGM1_Tests.KadMed_TrueEquinus()
    CGM1_Tests.KadMed_TrueEquinus_leftSkinMarkers()
