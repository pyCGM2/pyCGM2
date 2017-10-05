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
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums

class CGM1i_custom_calibrationTest():

    @classmethod
    def harrigton_fullPredictor(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington()

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_har", useRightHJCnode="RHJC_har").compute()


        # tests
        #  - altered HJC is explained firsly in the technical frame with the suffix ( har, ...) and subsquently  consider with no suffix in the anatomical frame
        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("RHJC").m_local, decimal=5)

        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("LHJC").m_local, decimal=5)

    @classmethod
    def harrigton_pelvisWidthPredictor(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

         # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington(predictors=pyCGM2Enums.HarringtonPredictor.PelvisWidth)

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_har", useRightHJCnode="RHJC_har").compute()


        # tests
        #  - altered HJC is explained firsly in the technical frame with the suffix ( har, ...) and subsquently  consider with no suffix in the anatomical frame
        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("RHJC").m_local, decimal=5)

        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("LHJC").m_local, decimal=5)


    @classmethod
    def harrigton_legLengthPredictor(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

         # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).harrington(predictors=pyCGM2Enums.HarringtonPredictor.LegLength)

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_har", useRightHJCnode="RHJC_har").compute()


        # tests
        #  - altered HJC is explained firsly in the technical frame with the suffix ( har, ...) and subsquently  consider with no suffix in the anatomical frame
        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("RHJC").m_local, decimal=5)

        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_har").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("LHJC").m_local, decimal=5)



    @classmethod
    def customLocalPosition(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).custom(position_Left = np.array([1,2,3]),
                                                  position_Right = np.array([1,2,3]), methodDesc = "us") # add node to pelvis

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_us", useRightHJCnode="RHJC_us").compute()


        # tests
        #  - altered HJC is explained firsly in the technical frame with the suffix ( har, ...) and subsquently  consider with no suffix in the anatomical frame
        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_us").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("RHJC").m_local, decimal=5)

        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_us").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("LHJC").m_local, decimal=5)



    @classmethod
    def hara_regressions(cls):
        """
        GOAL : compare Joint centres and foot Offset

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"


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
        # initial
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_hara", useRightHJCnode="RHJC_hara").compute()

        btkTools.smartWriter(acqStatic, "outStatic_Hara.c3d")

        # tests
        #  - altered HJC is explained firsly in the technical frame with the suffix ( har, ...) and subsquently  consider with no suffix in the anatomical frame
        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("RHJC_hara").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("RHJC").m_local, decimal=5)

        np.testing.assert_almost_equal(model.getSegment("Pelvis").getReferential("TF").static.getNode_byLabel("LHJC_hara").m_local,
                                      model.getSegment("Pelvis").anatomicalFrame.static.getNode_byLabel("LHJC").m_local, decimal=5)


    @classmethod
    def basicCGM1_BodyBuilderFoot(cls):  #def basicCGM1(self):
        """
        goal : know  the differenece on foot offset of a foot referential built according a sequence metionned in some bodybuilder code:
        LFoot = [LTOE,LAJC-LTOE,LAJC-LKJC,zyx]

        """
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            useBodyBuilderFoot=True).compute()

        spf_l,sro_l= model.getViconFootOffset("Left")
        spf_r,sro_r= model.getViconFootOffset("Right")

        btkTools.smartWriter(acqStatic, "CGM1_calibrationTest-basicCGM1.c3d")
        # TESTS ------------------------------------------------

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


        # foot offsets
        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(spf_l,vicon_spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(spf_r,vicon_spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(sro_l,vicon_sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs bodyBuilderFoot (%.6f)" %(sro_r,vicon_sro_r))

#        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
#        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
#        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
#        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

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
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionOFF()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationOFF_tibialTorsionON()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationOFF_shankRotationON_tibialTorsionOFF()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationON_tibialTorsionOFF()
    CGM1_calibrationTest.basicCGM1_manualOffset_thighRotationON_shankRotationON_tibialTorsionON()
    logging.info("######## PROCESS CGM1 --> Done ######")

#    logging.info("######## PROCESS CGM 1.1 --- MANUAL ######")
#    CGM11_calibrationTest.basicCGM1_manualOffsets() # work
#    CGM11_calibrationTest.basicCGM1_manualThighShankRotation() # work
#    CGM11_calibrationTest.basicCGM1_manualTibialTorsion() # work
#    CGM11_calibrationTest.advancedCGM1_kadMed_manualTibialTorsion() # work
#    logging.info("######## PROCESS CGM 1.1 --- MANUAL --> Done ######")


#
#    # CGM1 - custom
#    logging.info("######## PROCESS custom CGM1 ######")
#    CGM1i_custom_calibrationTest.harrigton_fullPredictor()
#    CGM1i_custom_calibrationTest.harrigton_pelvisWidthPredictor()
#    CGM1i_custom_calibrationTest.harrigton_legLengthPredictor()
#
#    CGM1i_custom_calibrationTest.customLocalPosition()
#    CGM1i_custom_calibrationTest.hara_regressions()
#    CGM1i_custom_calibrationTest.basicCGM1_BodyBuilderFoot() # not really a test
#    logging.info("######## PROCESS custom CGM1 --> Done ######")
