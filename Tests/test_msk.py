# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# pytest -s --disable-pytest-warnings  test_msk.py::Test_MSK_lowLevel::test_cgm23_scaling

import matplotlib.pyplot as plt

import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2


from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.Model import  modelFilters,modelDecorator
from pyCGM2 import enums
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Model.Opensim import osimProcessing


class Test_MSK_lowLevel:
    def test_cgm23_scaling(self):

        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitModels\CGM2.3\\Hannibal-medial\\"

        staticFilename = "static.c3d"
        gaitFilename= "gait1.c3d"

        markerDiameter=14
        required_mp={
        'Bodymass'   : 71.0,
        'Height'   : 1780.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
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

        # --- Calibration ---
        # ---check marker set used----
        acqStatic = btkTools.smartReader(DATA_PATH +  staticFilename)
        acqStatic =  btkTools.applyTranslators(acqStatic,translators)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        # # ---check marker set used----
        # dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        #
        # # --------------------------MODEL--------------------------------------
        # # ---definition---
        # model=cgm2.CGM2_3()
        # model.configure(detectedCalibrationMethods=dcm)
        # model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        #
        # if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD: actual_trackingMarkers.append("LKNE")
        # if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD: actual_trackingMarkers.append("RKNE")
        # model.setStaticTrackingMarkers(actual_trackingMarkers)
        #
        # # --------------------------STATIC CALBRATION--------------------------
        # scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure
        #
        # # ---initial calibration filter----
        # # use if all optional mp are zero
        # modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
        #                                     leftFlatFoot = True, rightFlatFoot = True,
        #                                     headFlat= True,
        #                                     markerDiameter=14.0,
        #                                     ).compute()
        #
        # # ---- Decorators -----
        # decorators.applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter)
        # decorators.applyHJCDecorators(model,hjcMethod)
        #
        #
        # # ----Final Calibration filter if model previously decorated -----
        # if model.decoratedModel:
        #     # initial static filter
        #     modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
        #                        leftFlatFoot = True, rightFlatFoot = True,
        #                        headFlat= True,
        #                        markerDiameter=14.0).compute()
        #
        #
        # # ----------------------CGM MODELLING----------------------------------
        # # ----motion filter----
        # modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Sodervisk,
        #                                           markerDiameter=markerDiameter)
        #
        # modMotion.compute()





        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model=cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)


        # ---- Calibration ----
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           markerDiameter=markerDiameter).compute()


        # ------- OPENSIM IK --------------------------------------

        acqGait = btkTools.smartReader(DATA_PATH +  gaitFilename)
        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)

        # --- osim builder ---
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/pyCGM2/Settings/opensim/models/cgm2-msk/cgm2_3-markerset.xml" #pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"

        # --- old implementation - No Scaling
        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                DATA_PATH)
        oscf.addMarkerSet(markersetFile)
        scalingOsim1 = oscf.build(exportOsim=False)



        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"

        osrf1 = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim1,
                                                          cgmFittingProcedure,
                                                          DATA_PATH,
                                                          acqGait )

        acqIK1 = osrf1.run(str(DATA_PATH + gaitFilename ),exportSetUp=False)



        modMotion_ik1=modelFilters.ModelMotionFilter(scp,acqIK1,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik1.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK1)
        finalJcs.compute(description="ik1", pointLabelSuffix = "ik1")#

        # btkTools.smartWriter(acqIK,"cgm23_fullIK_Motion.c3d")

        # --- NEW implementation - WITH Scaling
        osimProcessing.smartTrcExport(acqStatic,DATA_PATH +  "staticVerif")

        scaleToolFile = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/pyCGM2/Settings/opensim/models/cgm2-msk/CGM23_scaling_setup.xml"
        statictrc = "C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/pyCGM2/Settings/opensim/models/cgm2-msk/staticVerif.trc"  #DATA_PATH + "staticVerif.trc"

        osf = opensimFilters.opensimScalingFilter(osimfile,markersetFile,scaleToolFile,statictrc,"xml_output.xml",required_mp)
        scalingOsim2 = osf.getScaledModel()


        osrf2 = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim2,
                                                          cgmFittingProcedure,
                                                          DATA_PATH,
                                                          acqGait )


        acqIK2 = osrf2.run(str(DATA_PATH + gaitFilename ),exportSetUp=False)


        modMotion_ik2=modelFilters.ModelMotionFilter(scp,acqIK2,model,enums.motionMethod.Sodervisk,
                                                    useForMotionTest=True)
        modMotion_ik2.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK2)
        finalJcs.compute(description="ik2", pointLabelSuffix = "ik2")#




        fig = plt.figure(figsize=(10,4), dpi=100,facecolor="white")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)

        ax1.plot(acqIK1.GetPoint("LKneeAngles_ik1").GetValues()[:,0],"-r")
        ax1.plot(acqIK2.GetPoint("LKneeAngles_ik2").GetValues()[:,0],"-ob")

        ax2.plot(acqIK1.GetPoint("LKneeAngles_ik1").GetValues()[:,1],"-r")
        ax2.plot(acqIK2.GetPoint("LKneeAngles_ik2").GetValues()[:,1],"-ob")

        ax3.plot(acqIK1.GetPoint("LKneeAngles_ik1").GetValues()[:,2],"-r")
        ax3.plot(acqIK2.GetPoint("LKneeAngles_ik2").GetValues()[:,2],"-ob")

        plt.show()
        # import ipdb; ipdb.set_trace()
