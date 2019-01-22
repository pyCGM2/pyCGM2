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

class CGM1_calibrationTest():

    @classmethod
    def basicCGM1(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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

        #btkTools.smartWriter(acqStatic, "CGM1_calibrationTest-basicCGM1.c3d")

        # TESTS ------------------------------------------------
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


    @classmethod
    def basicCGM1_flatFoot(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, leftFlatFoot = True, rightFlatFoot = True).compute()

        # TESTS ------------------------------------------------
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)



    @classmethod
    def basicCGM1_soleDelta_FlatFoot(cls):  # notice : soleDelta has no influence with flatfoot disable

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-SoleDelta-FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        'LeftSoleDelta' : 10,
        'RightSoleDelta' : 10,
        }
        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = True, rightFlatFoot = True).compute()

        #btkTools.smartWriter(acqStatic, "CGM1_calibrationTest-basicCGM1.c3d")
        # TESTS ------------------------------------------------
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)

    @classmethod
    def basicCGM1_manualTibialTorsion(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            viconCGM1compatible=True).compute()


        # TESTS ------------------------------------------------
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        # nodes
        # np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        # np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")
        #
        # np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        # np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)

        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_manualTibial.c3d")

    @classmethod
    def basicCGM1_manualTibialTorsion_withDecorator(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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


        # TESTS ------------------------------------------------
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        # nodes
        # np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        # np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")
        #
        # np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"Chord")
        # np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"Chord")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)

        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_manualTibial.c3d")


    @classmethod
    def basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionOFF(cls):
        """
        CGM1 manual offset behaviour :
        => - Manual Thigh Rotation must modify ShankRotation.
           - If zero TibialTorsion inputs -> flag RightTibialTorsion must be off and TibialTorsion keep zero value
        """


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-manualOffsets-thiON_shaOFF_torsionOFF\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        'LeftThighRotation' : 10,
        'LeftShankRotation' : 0,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 10,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 0
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # decorators
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",model.mp["LeftThighRotation"],markerDiameter,model.mp["LeftTibialTorsion"],model.mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",model.mp["RightThighRotation"],markerDiameter,model.mp["RightTibialTorsion"],model.mp["RightShankRotation"])

        # finql calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,viconCGM1compatible=True  ).compute()

        # TESTS
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"manual ThighOffset")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"manual ThighOffset")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"manual ThighOffset")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"manual ThighOffset")

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )


    @classmethod
    def basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionON(cls):
        """
        CGM1 manual offset behaviour :
        => - Manual Thigh Rotation must modify ShankRotation.
           - If zero TibialTorsion inputs -> flag RightTibialTorsion must be off and TibialTorsion keep zero value
        """


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-manualOffsets-thiON_shaOFF_torsionON\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        'LeftThighRotation' : 10,
        'LeftShankRotation' : 0,
        'LeftTibialTorsion' : -30.0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 10,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : -30.0
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # decorators
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",model.mp["LeftThighRotation"],markerDiameter,model.mp["LeftTibialTorsion"],model.mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",model.mp["RightThighRotation"],markerDiameter,model.mp["RightTibialTorsion"],model.mp["RightShankRotation"])

        # finql calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,viconCGM1compatible=True  ).compute()

        # TESTS
        # offset testing
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"manual ThighOffset")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"manual ThighOffset")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"manual ThighOffset")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"manualTHIoffset-manualTT")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"manualTHIoffset-manualTT")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"manualTHIoffset-manualTT")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"manualTHIoffset-manualTT")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

    @classmethod
    def basicCGM1_manualOffset_thighRotationOFF_shankRotationON_tibialTorsionOFF(cls):
        """
        CGM1 manual offset behaviour :
        => - Manual Thigh Rotation must modify ShankRotation.
           - If zero TibialTorsion inputs -> flag RightTibialTorsion must be off and TibialTorsion keep zero value
        """


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-manualOffsets-thiOFF_shaON_torsionOFF\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        'LeftShankRotation' : -10,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : -10,
        'RightTibialTorsion' : 0
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # decorators
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",model.mp["LeftThighRotation"],markerDiameter,model.mp["LeftTibialTorsion"],model.mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",model.mp["RightThighRotation"],markerDiameter,model.mp["RightTibialTorsion"],model.mp["RightShankRotation"])

        # finql calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model ).compute()

        # TESTS
        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

        np.testing.assert_almost_equal(model.mp_computed["LeftShankRotationOffset"] ,0, decimal = 3)
        np.testing.assert_almost_equal(model.mp_computed["RightShankRotationOffset"] ,0, decimal = 3)



    @classmethod
    def basicCGM1_manualOffset_thighRotationOFF_shankRotationON_tibialTorsionON(cls):
        """
        CGM1 manual offset behaviour :
        => - Manual Thigh Rotation must modify ShankRotation.
           - If zero TibialTorsion inputs -> flag RightTibialTorsion must be off and TibialTorsion keep zero value
        """


        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic-manualOffsets-thiOFF_shaON_torsionON\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
        'LeftShankRotation' : -10,
        'LeftTibialTorsion' : -30,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : -10,
        'RightTibialTorsion' : -30
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # decorators
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",model.mp["LeftThighRotation"],markerDiameter,model.mp["LeftTibialTorsion"],model.mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",model.mp["RightThighRotation"],markerDiameter,model.mp["RightTibialTorsion"],model.mp["RightShankRotation"])

        # finql calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model ).compute()

        # TESTS
        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        np.testing.assert_almost_equal(model.mp_computed["LeftShankRotationOffset"] ,0, decimal = 3)
        np.testing.assert_almost_equal(model.mp_computed["RightShankRotationOffset"] ,0, decimal = 3)


    @classmethod
    def advancedCGM1_kad_noOptions(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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

        # final calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # TESTS ------------------------------------------------
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


    @classmethod
    def advancedCGM1_kad_flatFoot(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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
                                   leftFlatFoot = True, rightFlatFoot = True).compute()


        # TESTS ------------------------------------------------
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD")

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-Med\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        # TESTS ------------------------------------------------
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)

        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_midMaleolus.c3d")




    @classmethod
    def advancedCGM1_kad_midMaleolus_markerDiameter(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-Med-markerDiameter\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
        model.configure()

        markerDiameter=25.0
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,markerDiameter=markerDiameter,
                                            leftFlatFoot = False, rightFlatFoot = False).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic,markerDiameter=markerDiameter, side="both")


        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                             markerDiameter=markerDiameter).compute()


        # TESTS ------------------------------------------------
        offsetTesting(acqStatic,model,display = True, unitTesting=True)

        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"mid")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"mid")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)

        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_midMaleolus_markerDiameter.c3d")

    @classmethod
    def advancedCGM1_kad_manualTibialTorsion(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-manualTibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1
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

        # TESTS ------------------------------------------------
        # offset testing
        offsetTesting(acqStatic,model,display =True, unitTesting=True)


        # node description
        np.testing.assert_equal(model.getSegment("Left Thigh").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Thigh").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")

        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LKJC").m_desc ,"KAD")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RKJC").m_desc ,"KAD")


        np.testing.assert_equal(model.getSegment("Left Shank").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD-manualTT")
        np.testing.assert_equal(model.getSegment("Right Shank").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD-manualTT")

        np.testing.assert_equal(model.getSegment("Left Foot").getReferential("TF").static.getNode_byLabel("LAJC").m_desc ,"KAD-manualTT")
        np.testing.assert_equal(model.getSegment("Right Foot").getReferential("TF").static.getNode_byLabel("RAJC").m_desc ,"KAD-manualTT")

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)



        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_manualTibial.c3d")







if __name__ == "__main__":


    # CGM 1
    logging.info("######## PROCESS CGM1 ######")

    CGM1_calibrationTest.basicCGM1()
    CGM1_calibrationTest.basicCGM1_flatFoot()
    CGM1_calibrationTest.basicCGM1_soleDelta_FlatFoot()

    CGM1_calibrationTest.advancedCGM1_kad_noOptions()
    CGM1_calibrationTest.advancedCGM1_kad_flatFoot()
    CGM1_calibrationTest.advancedCGM1_kad_midMaleolus()
    CGM1_calibrationTest.advancedCGM1_kad_midMaleolus_markerDiameter()

    CGM1_calibrationTest.advancedCGM1_kad_manualTibialTorsion()
    CGM1_calibrationTest.basicCGM1_manualTibialTorsion()

    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionOFF()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionON()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationOFF_shankRotationON_tibialTorsionOFF()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationOFF_shankRotationON_tibialTorsionON()


    logging.info("######## PROCESS CGM1 --> Done ######")
