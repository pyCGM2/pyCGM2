# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pdb
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame
from pyCGM2.Model.CGM2 import cgm
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric


class CGM1_GaitPatternFullAnglesTest():

    @classmethod
    def TrueEquinus_S1_static(cls):
        """

        """

        MAIN_PATH = pyCGM2.CONFIG.MAIN_BENCHMARK_PATH + "True equinus\S01\PIG\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        markerDiameter = 25.0
        leftFlatFoot_flag = False
        rightFlatFoot_flag = False

        required_mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665.0,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.0,
        'LeftSoleDelta' : 0.0,
        'RightSoleDelta' : 0.0,
        }
        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftTibialTorsion' : 0 ,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightTibialTorsion' : 0 ,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        }


        # ---definition---
        model=cgm.CGM1LowerLimbs()
        model.configure()
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)


        # ---check marker set used----
        staticMarkerConfiguration= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)


        # --------------------------STATIC CALBRATION--------------------------
        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        # ---initial calibration filter----
        # use if all optional mp are zero
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = leftFlatFoot_flag, rightFlatFoot = rightFlatFoot_flag,
                                            markerDiameter=markerDiameter,
                                            ).compute()
#        # ---- Decorators -----
#        # Goal = modified calibration according the identified marker set or if offsets manually set
#
#        # initialisation of node label
#        useLeftKJCnodeLabel = "LKJC_chord"
#        useLeftAJCnodeLabel = "LAJC_chord"
#        useRightKJCnodeLabel = "RKJC_chord"
#        useRightAJCnodeLabel = "RAJC_chord"
#
#        # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
#        #   This
#        if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
#            logging.warning("CASE FOUND ===> Left Side - CGM1 - Origine - manual offsets")
#            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
#            useLeftKJCnodeLabel = "LKJC_mo"
#            useLeftAJCnodeLabel = "LAJC_mo"
#
#
#        if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
#            logging.warning("CASE FOUND ===> Right Side - CGM1 - Origine - manual offsets")
#            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
#            useRightKJCnodeLabel = "RKJC_mo"
#            useRightAJCnodeLabel = "RAJC_mo"
#
#        # case 2 : kad FOUND and NO medial Ankle
#        if staticMarkerConfiguration["leftKadFlag"]:
#            logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD variant")
#            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left", displayMarkers = False)
#            useLeftKJCnodeLabel = "LKJC_kad"
#            useLeftAJCnodeLabel = "LAJC_kad"
#        if staticMarkerConfiguration["rightKadFlag"]:
#            logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
#            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right", displayMarkers = False)
#            useRightKJCnodeLabel = "RKJC_kad"
#            useRightAJCnodeLabel = "RAJC_kad"
#
#        # case 3 : both kad and medial ankle FOUND
#        if staticMarkerConfiguration["leftKadFlag"]:
#            if staticMarkerConfiguration["leftMedialAnkleFlag"]:
#                logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD + medial ankle ")
#                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
#                useLeftAJCnodeLabel = "LAJC_mid"
#
#        if staticMarkerConfiguration["rightKadFlag"]:
#            if staticMarkerConfiguration["rightMedialAnkleFlag"]:
#                logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD + medial ankle ")
#                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
#                useRightAJCnodeLabel = "RAJC_mid"

#        # ----Final Calibration filter if model previously decorated -----
#        if model.decoratedModel:
#            # initial static filter
#            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
#                               useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
#                               useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
#                               markerDiameter=markerDiameter).compute()


        # TESTS------------

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)

        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)




        # tibial rotation
        ltt_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LTibialTorsion").value().GetInfo().ToDouble()[0])
        rtt_vicon =np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RTibialTorsion").value().GetInfo().ToDouble()[0])

        logging.info(" LTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"]))
        logging.info(" RTibialTorsion : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"]))


        # foot offsets
        spf_l,sro_l = model.getViconFootOffset("Left")
        spf_r,sro_r = model.getViconFootOffset("Right")
        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        logging.info(" LStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_spf_l,spf_l))
        logging.info(" RStaticPlantFlex : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_spf_r,spf_r))
        logging.info(" LStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_sro_l,sro_l))
        logging.info(" RStaticRotOff : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(vicon_sro_r,sro_r))

         # thigh and shank Offsets
        lto = model.getViconThighOffset("Left")
        lso = model.getViconShankOffset("Left")
        rto = model.getViconThighOffset("Right")
        rso = model.getViconShankOffset("Right")

        lto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0])
        lso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0])

        rto_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0])
        rso_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0])

        logging.info(" LThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lto_vicon,lto))
        logging.info(" LShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(lso_vicon,lso))
        logging.info(" RThighRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rto_vicon,rto))
        logging.info(" RShankRotation : Vicon (%.6f)  Vs pyCGM2 (%.6f)" %(rso_vicon,rso))

        # tests on offsets
        np.testing.assert_almost_equal(-ltt_vicon,model.mp_computed["LeftTibialTorsionOffset"] , decimal = 2)
        np.testing.assert_almost_equal(rtt_vicon,model.mp_computed["RightTibialTorsionOffset"] , decimal = 3) # FAIL: -19.663185714739587 instead -19.655711786374621

        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)




        # ------ Test 1 Motion Axe X -------
        gaitFilename="static.c3d"   #"staticComparisonPipelines.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))


        # Motion FILTER
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 markerDiameter=markerDiameter)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")


        # absolute angles
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      eulerSequences=["TOR","TOR", "TOR"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")
#
#
#
#        # output and plot
        btkTools.smartWriter(acqGait, "test_gaitPattern.c3d")

        plt.figure()
        plt.title("knee angles")
        plt.plot(acqGait.GetPoint("LHipAngles").GetValues()[:,0])
        plt.plot(acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues()[:,0],"-r")
        #plt.plot(acqGait.GetPoint("LHipAnglesPIG").GetValues()[:,0],"-g")

        plt.figure()
        plt.title("ankle angles")
        plt.plot(acqGait.GetPoint("LAnkleAngles").GetValues()[:,1])
        plt.plot(acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues()[:,1],"-r")

        plt.figure()
        plt.title("foot progress angles")
        plt.plot(acqGait.GetPoint("LFootProgressAngles").GetValues()[:,2])
        plt.plot(acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues()[:,2],"-r")


#        # tests on joint angles
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
#                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
#                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
#                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
#                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
#                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
#        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
#                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
#
#
#        #tests influence by vicon error
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =2)
#
#        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =2)
#        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
#                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =2)



if __name__ == "__main__":

    plt.close("all")
    CGM1_GaitPatternFullAnglesTest.TrueEquinus_S1_static()
