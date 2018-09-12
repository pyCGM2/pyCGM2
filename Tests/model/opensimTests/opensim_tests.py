# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 14:04:13 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt

import pyCGM2

from pyCGM2 import btk
import pdb

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# pyCGM2
from pyCGM2.Tools import  btkTools

from pyCGM2.Model.CGM2 import cgm2Julie
from pyCGM2.Model import  modelFilters,modelDecorator

from pyCGM2.Model.Opensim import osimProcessing,opensimFilters
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric



def comparisonOpensimVsCGM(motFilename,acqIK,pointLabelSuffix):
    # note : assert_almost_equal tests fail. Prefer rmsd computation


    values = osimProcessing.mot2pointValues(motFilename,["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"], orientation =[1.0,1.0,1.0])
    myValues = acqIK.GetPoint(str("RHipAngles"+"_" +pointLabelSuffix)).GetValues()
    #np.testing.assert_almost_equal( values,myValues, decimal =3)
    np.testing.assert_array_less(numeric.rms((values-myValues), axis = 0), 0.05)

    values = osimProcessing.mot2pointValues(motFilename,["knee_flexion_r", "knee_adduction_r", "knee_rotation_r"], orientation =[-1.0,1.0,1.0])
    myValues = acqIK.GetPoint(str("RKneeAngles"+"_" +pointLabelSuffix)).GetValues()
    #np.testing.assert_almost_equal( values,myValues, decimal =3)
    np.testing.assert_array_less(numeric.rms((values-myValues), axis = 0), 0.05)

    values = osimProcessing.mot2pointValues(motFilename,["ankle_flexion_r", "ankle_rotation_r", "ankle_adduction_r"], orientation =[1.0,-1.0,1.0])
    myValues = acqIK.GetPoint(str("RAnkleAngles"+"_" +pointLabelSuffix)).GetValues()
    #np.testing.assert_almost_equal( values,myValues, decimal =3)
    np.testing.assert_array_less(numeric.rms((values-myValues), axis = 0), 0.05)

    values = osimProcessing.mot2pointValues(motFilename,["forefoot_flexion_r", "forefoot_adduction_r", "forefoot_rotation_r"], orientation =[1.0,1.0,-1.0])
    myValues = acqIK.GetPoint(str("RForeFootAngles"+"_" +pointLabelSuffix)).GetValues()
    #np.testing.assert_almost_equal( values,myValues, decimal =3)
    np.testing.assert_array_less(numeric.rms((values-myValues), axis = 0), 0.05)



class CGM2_openSimTest():

    @classmethod
    def kinematicFitting_oneFile_cgmProcedure(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\LowerLimb\\subject10_S1A1_julieDataSet_noModelOutputs\\"

        config=dict()
        config["static"] = "StaticR010S1A1.c3d"
        config["dynamicTrial"] =["R010S1A1Gait001.c3d",
                             "R010S1A1Gait003.c3d",
                             "R010S1A1Gait004.c3d",
                             "R010S1A1Gait005.c3d",
                             "R010S1A1Gait007.c3d",
                             "R010S1A1Gait010.c3d"]

        configMP = {"Bodymass" : 64.0 ,
                "LeftLegLength" : 865.0,
                "RightLegLength" : 855.0,
                "LeftKneeWidth" : 100.0,
                "RightKneeWidth" : 101.0,
                "LeftAnkleWidth" : 69.0,
                "RightAnkleWidth" : 69.0}

        requiredMarkers= ["LASI", "RASI","LPSI", "RPSI", "RTHIAP", "RTHIAD", "RTHI", "RKNE", "RSHN","RTIAP", "RTIB", "RANK", "RHEE", "RTOE","RCUN","RD1M","RD5M" ]


        acqStatic=btkTools.smartReader(DATA_PATH+config["static"])

        # model
        model=cgm2Julie.CGM2ModelInf()
        model.configure()
        markerDiameter=14


        mp={
        'Bodymass'   : configMP["Bodymass"],
        'LeftLegLength' : configMP["LeftLegLength"],
        'RightLegLength' : configMP["RightLegLength"] ,
        'LeftKneeWidth' : configMP["LeftKneeWidth"],
        'RightKneeWidth' : configMP["RightKneeWidth"],
        'LeftAnkleWidth' : configMP["LeftAnkleWidth"],
        'RightAnkleWidth' : configMP["RightAnkleWidth"],
        }

         #offset 2ndToe Joint
        toe = acqStatic.GetPoint("RTOE").GetValues()[:,:].mean(axis=0)
        mp["rightToeOffset"] =  (toe[2]-markerDiameter/2.0)/2.0

        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------

        #  --- Initial calibration
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   rightFlatHindFoot = True).compute() # Initial calibration

        # --- decorator
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, leftMedialKneeLabel="LKNM", rightMedialKneeLabel="RKNM")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, leftMedialAnkleLabel="LMMA", rightMedialAnkleLabel="RMMA")

        # --- Updated calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                       useLeftKJCnode="LKJC_mid",
                                       useRightKJCnode="RKJC_mid",
                                       useLeftAJCnode="LAJC_mid",
                                       useRightAJCnode="RAJC_mid",
                                       rightFlatHindFoot = True).compute()


        # ---- optional -update static c3d
        #btkTools.smartWriter(acqStatic, "StaticCalibrated.c3d")


        # -----------CGM MOTION  6dof--------------------

        # --- reader and checking
        acqGait = btkTools.smartReader(DATA_PATH + config["dynamicTrial"][0])
        btkTools.checkMarkers(acqGait,requiredMarkers)
        btkTools.isGap(acqGait,requiredMarkers)
        btkTools.checkFirstAndLastFrame(acqGait, "LASI")
        logging.info("dyn Acquisition ---> OK ")


        # --- filters
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()


        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="2_6dof")

        # --- writer
        #btkTools.smartWriter(acqGait, "motionCalibrated.c3d")


        # ------- OPENSIM IK --------------------------------------
        cgmprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\cgm2-markerset.xml"

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\cgm2-model.osim"


        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmprocedure)
        oscf.addMarkerSet(markersetFile)
        fittingOsim = oscf.build()

        filename = config["dynamicTrial"][0]
        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)

        cgmFittingProcedure.updateMarkerWeight("LASI",100)

        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\ikSetUp_template.xml"

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          fittingOsim,
                                                          cgmFittingProcedure,
                                                          DATA_PATH )
        acqIK = osrf.run(acqGait,str(DATA_PATH + filename ))

        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.setFilterBool(False)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#


        # recup du mot file
        motFilename = str(DATA_PATH + config["dynamicTrial"][0][:-4]+".mot")

        comparisonOpensimVsCGM(motFilename,acqIK,"2_ik")



    @classmethod
    def kinematicFitting_oneFile_generalProcedure(cls):

        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\LowerLimb\\subject10_S1A1_julieDataSet_noModelOutputs\\"

        config=dict()
        config["static"] = "StaticR010S1A1.c3d"
        config["dynamicTrial"] =["R010S1A1Gait001.c3d",
                             "R010S1A1Gait003.c3d",
                             "R010S1A1Gait004.c3d",
                             "R010S1A1Gait005.c3d",
                             "R010S1A1Gait007.c3d",
                             "R010S1A1Gait010.c3d"]

        configMP = {"Bodymass" : 64.0 ,
                "LeftLegLength" : 865.0,
                "RightLegLength" : 855.0,
                "LeftKneeWidth" : 100.0,
                "RightKneeWidth" : 101.0,
                "LeftAnkleWidth" : 69.0,
                "RightAnkleWidth" : 69.0}

        requiredMarkers= ["LASI", "RASI","LPSI", "RPSI", "RTHIAP", "RTHIAD", "RTHI", "RKNE", "RSHN","RTIAP", "RTIB", "RANK", "RHEE", "RTOE","RCUN","RD1M","RD5M" ]


        acqStatic=btkTools.smartReader(DATA_PATH+config["static"])

        # model
        model=cgm2Julie.CGM2ModelInf()
        model.configure()
        markerDiameter=14


        mp={
        'Bodymass'   : configMP["Bodymass"],
        'LeftLegLength' : configMP["LeftLegLength"],
        'RightLegLength' : configMP["RightLegLength"] ,
        'LeftKneeWidth' : configMP["LeftKneeWidth"],
        'RightKneeWidth' : configMP["RightKneeWidth"],
        'LeftAnkleWidth' : configMP["LeftAnkleWidth"],
        'RightAnkleWidth' : configMP["RightAnkleWidth"],
        }

         #offset 2ndToe Joint
        toe = acqStatic.GetPoint("RTOE").GetValues()[:,:].mean(axis=0)
        mp["rightToeOffset"] =  (toe[2]-markerDiameter/2.0)/2.0

        model.addAnthropoInputParameters(mp)

        # -----------CGM STATIC CALIBRATION--------------------

        #  --- Initial calibration
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   rightFlatHindFoot = True).compute() # Initial calibration

        # --- decorator
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, leftMedialKneeLabel="LKNM", rightMedialKneeLabel="RKNM")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, leftMedialAnkleLabel="LMMA", rightMedialAnkleLabel="RMMA")

        # --- Updated calibration
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                       useLeftKJCnode="LKJC_mid",
                                       useRightKJCnode="RKJC_mid",
                                       useLeftAJCnode="LAJC_mid",
                                       useRightAJCnode="RAJC_mid",
                                       rightFlatHindFoot = True).compute()


        # ---- optional -update static c3d
        #btkTools.smartWriter(acqStatic, "StaticCalibrated.c3d")


        # -----------CGM MOTION  6dof--------------------

        # --- reader and checking
        acqGait = btkTools.smartReader(DATA_PATH + config["dynamicTrial"][0])
        btkTools.checkMarkers(acqGait,requiredMarkers)
        btkTools.isGap(acqGait,requiredMarkers)
        btkTools.checkFirstAndLastFrame(acqGait, "LASI")
        logging.info("dyn Acquisition ---> OK ")


        # --- filters
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
        modMotion.compute()


        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="2_6dof")

        # --- writer
        #btkTools.smartWriter(acqGait, "motionCalibrated.c3d")


        # ------- OPENSIM IK --------------------------------------
        generalprocedure = opensimFilters.GeneralOpensimCalibrationProcedure()
        generalprocedure.setMarkers("Pelvis", ["LASI","RASI","LPSI","RPSI"])
        generalprocedure.setMarkers("Right Thigh", ["RKNE","RTHI","RTHIAP","RTHIAD"])
        generalprocedure.setMarkers("Right Shank", ["RANK","RTIB","RSHN","RTIAP"])
        generalprocedure.setMarkers("Right Hindfoot", ["RHEE","RCUN"])
        generalprocedure.setMarkers("Right Forefoot", ["RD1M","RD5M"])

        generalprocedure.setGeometry("hip_r","RHJC","Pelvis","Right Thigh")
        generalprocedure.setGeometry("knee_r","RKJC","Right Thigh","Right Shank")
        generalprocedure.setGeometry("ankle_r","RAJC","Right Shank","Right Hindfoot")
        generalprocedure.setGeometry("mtp_r","RvCUN","Right Hindfoot","Right Forefoot")

        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\cgm2-model.osim"
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\cgm2-markerset.xml"

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                generalprocedure)
        oscf.addMarkerSet(markersetFile)
        fittingOsim = oscf.build()

        filename = config["dynamicTrial"][0]
        generalMotionProcedure = opensimFilters.GeneralOpensimFittingProcedure()
        generalMotionProcedure.setMarkerWeight("LASI",100)
        generalMotionProcedure.setMarkerWeight("RASI",100)
        generalMotionProcedure.setMarkerWeight("LPSI",100)
        generalMotionProcedure.setMarkerWeight("RPSI",100)
        generalMotionProcedure.setMarkerWeight("LASI",100)
        generalMotionProcedure.setMarkerWeight("RTHI",100)
        generalMotionProcedure.setMarkerWeight("RTHIAP",100)
        generalMotionProcedure.setMarkerWeight("RTHIAD",100)
        generalMotionProcedure.setMarkerWeight("RKNE",100)
        generalMotionProcedure.setMarkerWeight("RTIB",100)
        generalMotionProcedure.setMarkerWeight("RTIAP",100)
        generalMotionProcedure.setMarkerWeight("RSHN",100)
        generalMotionProcedure.setMarkerWeight("RANK",100)
        generalMotionProcedure.setMarkerWeight("RHEE",100)
        generalMotionProcedure.setMarkerWeight("RCUN",100)
        generalMotionProcedure.setMarkerWeight("RD1M",100)
        generalMotionProcedure.setMarkerWeight("RD5M",100)

        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\draft-opensimPreProcessing\\ikSetUp_template.xml"

        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          fittingOsim,
                                                          generalMotionProcedure,
                                                          DATA_PATH )
        acqIK = osrf.run(acqGait,str(DATA_PATH + filename ))

        # -------- NEW MOTION FILTER ON IK MARKERS ------------------

        modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotion_ik.compute()

        finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
        finalJcs.setFilterBool(False)
        finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#


        # recup du mot file
        motFilename = str(DATA_PATH + config["dynamicTrial"][0][:-4]+".mot")

        comparisonOpensimVsCGM(motFilename,acqIK,"2_ik")

if __name__ == "__main__":

    logging.info("######## PROCESS CGM2 ######")
    CGM2_openSimTest.kinematicFitting_oneFile_cgmProcedure()
    #CGM2_openSimTest.kinematicFitting_oneFile_generalProcedure()
