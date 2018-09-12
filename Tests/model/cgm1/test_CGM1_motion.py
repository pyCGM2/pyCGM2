# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm
import pyCGM2.enums as pyCGM2Enums



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
    def basicCGM1(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # CALIBRATION
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,viconCGM1compatible=True).compute()

        print model.m_useRightTibialTorsion

        # --- Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        btkTools.smartWriter(acqGait, "test.c3d")


        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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



        # --- Test 2 Motion Axe -X ----
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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




    @classmethod
    def basicCGM1_flatFoot(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)


        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, leftFlatFoot = True, rightFlatFoot = True,
                                            viconCGM1compatible=True).compute()
        spf_l,sro_l = model.getViconFootOffset("Left")
        spf_r,sro_r = model.getViconFootOffset("Right")

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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


        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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


    @classmethod
    def basicCGM1_manualTibialTorsion(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0,
        'LeftTibialTorsion' : -30.0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : -30.0
        }
        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()
        # decorators
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",model.mp["LeftThighRotation"],markerDiameter,model.mp["LeftTibialTorsion"],model.mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",model.mp["RightThighRotation"],markerDiameter,model.mp["RightTibialTorsion"],model.mp["RightShankRotation"])

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()

        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        # --- Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,
                                                 viconCGM1compatible=True)
        modMotion.compute()

        btkTools.smartWriter(acqGait, "test.c3d")


        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()
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
        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)



        # --- Test 2 Motion Axe -X ----
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,
                                                 viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()
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
        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)


    @classmethod
    def advancedCGM1_kad_noOptions(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()



        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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

    @classmethod
    def advancedCGM1_kad_flatFoot(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()



        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True,
                                   viconCGM1compatible=True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
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


    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-Med\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

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

        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


#        #foot (Do not consider since Vicon Foot employs wrong shank axis)
#        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
#        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
#        np.testing.assert_almost_equal( R_leftFoot,
#                                        R_leftFootVicon, decimal =3)
#        np.testing.assert_almost_equal( R_rightFoot,
#                                        R_rightFootVicon, decimal =3)



        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

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

        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


#        #foot (Do not consider since Vicon Foot employs wrong shank axis)
#        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
#        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
#        np.testing.assert_almost_equal( R_leftFoot,
#                                        R_leftFootVicon, decimal =3)
#        np.testing.assert_almost_equal( R_rightFoot,
#                                        R_rightFootVicon, decimal =3)

    @classmethod
    def advancedCGM1_kad_manualTibialTorsion(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-manualTibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()

        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'LeftTibialTorsion' : -30.0,
        'RightTibialTorsion' : -30.0
        }
        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()


        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

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

        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


    #        #foot (Do not consider since Vicon Foot employs wrong shank axis)
    #        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
    #        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
    #        np.testing.assert_almost_equal( R_leftFoot,
    #                                        R_leftFootVicon, decimal =3)
    #        np.testing.assert_almost_equal( R_rightFoot,
    #                                        R_rightFootVicon, decimal =3)



        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 usePyCGM2_coordinateSystem=True,viconCGM1compatible=True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")

        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3)

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()

        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

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

        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


    #        #foot (Do not consider since Vicon Foot employs wrong shank axis)
    #        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY")
    #        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY")
    #        np.testing.assert_almost_equal( R_leftFoot,
    #                                        R_leftFootVicon, decimal =3)
    #        np.testing.assert_almost_equal( R_rightFoot,
    #                                        R_rightFootVicon, decimal =3)





if __name__ == "__main__":

    logging.info("######## PROCESS CGM1 ######")
    CGM1_motionTest.basicCGM1()
    CGM1_motionTest.basicCGM1_flatFoot()
    CGM1_motionTest.basicCGM1_manualTibialTorsion()
    CGM1_motionTest.advancedCGM1_kad_noOptions()
    CGM1_motionTest.advancedCGM1_kad_flatFoot()
    CGM1_motionTest.advancedCGM1_kad_midMaleolus()
    CGM1_motionTest.advancedCGM1_kad_manualTibialTorsion()
    logging.info("######## PROCESS CGM1 --> Done######")
