# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

import pyCGM2

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric



def plotComparison(acq,acqs,angleLabel):
    fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
    fig.suptitle(angleLabel+"(red: StaticPIG - blue: dynamicPIC - green:FittingCGM)")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)


    ax1.plot(acqs.GetPoint(angleLabel).GetValues()[:,0],"-r")
    ax1.plot(acq.GetPoint(angleLabel).GetValues()[:,0],"-b")
    ax1.plot(acqs.GetPoint(angleLabel+"_cgm1_6dof").GetValues()[:,0],"-g")

    ax2.plot(acqs.GetPoint(angleLabel).GetValues()[:,1],"-r")
    ax2.plot(acq.GetPoint(angleLabel).GetValues()[:,1],"-b")
    ax2.plot(acqs.GetPoint(angleLabel+"_cgm1_6dof").GetValues()[:,1],"-g")

    ax3.plot(acqs.GetPoint(angleLabel).GetValues()[:,2],"-r")
    ax3.plot(acq.GetPoint(angleLabel).GetValues()[:,2],"-b")
    ax3.plot(acqs.GetPoint(angleLabel+"_cgm1_6dof").GetValues()[:,2],"-g")

    plt.show()


class CGM1_calibrationTest():

    @classmethod
    def basicCGM1(cls):
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\basic_static_StaticVsDynamicAngles\\"


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

        spf_l,sro_l= model.getViconFootOffset("Left")
        spf_r,sro_r= model.getViconFootOffset("Right")



        # TESTS ------------------------------------------------

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

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


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_l,vicon_spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_r,vicon_spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_l,vicon_sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_r,vicon_sro_r))

        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

        # -------- CGM FITTING -------------------------------------------------

        # ---- on c3d processed vicon static-pig operation
        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                                 markerDiameter=markerDiameter,
                                                 pigStatic=True,
                                                 viconCGM1compatible=False)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqStatic, "test_basicCGM1_staticAngles.c3d")


        # ---- on c3d processed vicon dynamic-pig operation

        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 02-dynamics.c3d"   #"staticComparisonPipelines.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))



#        # output and plot
        #btkTools.smartWriter(acqGait, "test_basicCGM1_staticAngles.c3d")
        plotComparison(acqGait,acqStatic,"LHipAngles")
        plotComparison(acqGait,acqStatic,"LKneeAngles")
        plotComparison(acqGait,acqStatic,"LAnkleAngles")

        plt.figure()
        plt.plot(acqStatic.GetPoint("LAnkleAngles").GetValues()[:,0] -acqStatic.GetPoint("LAnkleAngles"+"_cgm1_6dof").GetValues()[:,0])
        plt.figure()
        plt.plot(acqStatic.GetPoint("LAnkleAngles").GetValues()[:,2] -acqStatic.GetPoint("LAnkleAngles"+"_cgm1_6dof").GetValues()[:,2])


    @classmethod
    def basicCGM1_KAD(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic_static_StaticVsDynamicAngles\\"

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


#        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
#
#        # final calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


#        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
#                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_kad",
#                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_kad").compute()


        spf_l,sro_l= model.getViconFootOffset("Left")
        spf_r,sro_r= model.getViconFootOffset("Right")

        # TESTS ------------------------------------------------

        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

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


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_l,vicon_spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_r,vicon_spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_l,vicon_sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_r,vicon_sro_r))

        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

        # -------- CGM FITTING -------------------------------------------------


        # ---- on c3d processed vicon static-pig operation
        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                                 markerDiameter=markerDiameter,
                                                 pigStatic=True,
                                                 useLeftKJCmarker="LKJC_KAD", useLeftAJCmarker="LAJC_KAD",
                                                 useRightKJCmarker="RKJC_KAD", useRightAJCmarker="RAJC_KAD",
                                                 viconCGM1compatible=False)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqStatic, "test_basicCGM1_KAD-staticAngles.c3d")

        # ---- on c3d processed vicon dynamic-pig operation

        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 02-dynamics.c3d"   #"staticComparisonPipelines.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


#        # output and plot
#        btkTools.smartWriter(acqGait, "test_basicCGM1_KAD-staticAngles.c3d")

        plotComparison(acqGait,acqStatic,"LHipAngles")
        plotComparison(acqGait,acqStatic,"LKneeAngles")
        plotComparison(acqGait,acqStatic,"LAnkleAngles")

    @classmethod
    def basicCGM1_KAD_tibialTorsion(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-tibialTorsion-static_StaticVsDynamicAngles\\"

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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = False, rightFlatFoot = False).compute()


        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        # final calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = False, rightFlatFoot = False).compute()


        spf_l,sro_l= model.getViconFootOffset("Left")
        spf_r,sro_r= model.getViconFootOffset("Right")

        # TESTS ------------------------------------------------

        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

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


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_l,vicon_spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(spf_r,vicon_spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_l,vicon_sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(sro_r,vicon_sro_r))

        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

        # -------- CGM FITTING -------------------------------------------------

        # ---- on c3d processed vicon static-pig operation
        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Determinist,
                                                 markerDiameter=markerDiameter,
                                                 pigStatic=True,
                                                 useRightKJCmarker="RKJC_KAD", useRightAJCmarker="RAJC_KAD",
                                                 useLeftKJCmarker="LKJC_KAD", useLeftAJCmarker="LAJC_MID",
                                                 viconCGM1compatible = False)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")

        btkTools.smartWriter(acqStatic, "test_basicCGM1_KADmed-staticAngles.c3d")


        # ---- on c3d processed vicon dynamic-pig operation
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 02 -dynamic.c3d"   #"staticComparisonPipelines.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel


        # output and plot
        #btkTools.smartWriter(acqGait, "test_basicCGM1_KADmed-staticAngles.c3d")

        plotComparison(acqGait,acqStatic,"LHipAngles")
        plotComparison(acqGait,acqStatic,"LKneeAngles")
        plotComparison(acqGait,acqStatic,"LAnkleAngles")



if __name__ == "__main__":

    plt.close("all")

    # CGM 1
    logging.info("######## PROCESS CGM1 ######")
    CGM1_calibrationTest.basicCGM1()
#    CGM1_calibrationTest.basicCGM1_KAD()
#    CGM1_calibrationTest.basicCGM1_KAD_tibialTorsion()
