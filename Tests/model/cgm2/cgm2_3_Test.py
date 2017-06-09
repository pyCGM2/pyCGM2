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
from pyCGM2.Model.CGM2 import cgm2, modelFilters, modelDecorator
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Math import numeric
from pyCGM2.Model.Opensim import opensimFilters
import json
from collections import OrderedDict


if __name__ == "__main__":



    MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\"
    staticFilename = "static.c3d"
    gaitFilename= "gait trial 01.c3d"
    markerDiameter=14     
    mp={
    'Bodymass'   : 36.9,                
    'LeftLegLength' : 665.0,
    'RightLegLength' : 655.0 ,
    'LeftKneeWidth' : 102.7,
    'RightKneeWidth' : 102.0,
    'LeftAnkleWidth' : 64.5,
    'RightAnkleWidth' : 63.0,     
    'LeftSoleDelta' : 0,
    'RightSoleDelta' : 0,            
    }    

    CONTENT_INPUTS_CGM2_3 ="""
        {
        "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RTHIAP":"RTHAP",
            "RTHIAD":"RTHAD",
            "RTIB":"RTIBL",
            "RANK":"",
            "RTIBAP":"RTIAP",
            "RTIBAD":"RTIAD",
            "RHEE":"",
            "RTOE":"",
            "LTHI":"LTHL",
            "LKNE":"",
            "LTHIAP":"LTHAP",
            "LTHIAD":"LTHAD",
            "LTIB":"LTIBL",
            "LANK":"",
            "LTIBAP":"LTIAP",
            "LTIBAD":"LTIAD",
            "LHEE":"",
            "LTOE":""
            }
        }
      """        
    

    # --- Calibration ---                          
    acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
    
    model=cgm2.CGM2_3LowerLimbs()
    model.configure()        

    inputs = json.loads(CONTENT_INPUTS_CGM2_3,object_pairs_hook=OrderedDict)
    translators = inputs["Translators"]
    

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)    
    
    btkTools.smartWriter(acqStatic, "calibration2.c3d")
    
    model.addAnthropoInputParameters(mp)
                                
    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 


#    # cgm decorator
    modelDecorator.HipJointCenterDecorator(model).hara()  
#        
#    # final
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_Hara", useRightHJCnode="RHJC_Hara").compute()

    btkTools.smartWriter(acqStatic, "calibration.c3d")

    # ------ Fitting -------      
    acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

    acqGait =  btkTools.applyTranslators(acqGait,translators)     
    # Motion FILTER 

    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist)
    modMotion.compute()

    # relative angles
    modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
    
    # absolute angles 
    longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqGait,["LASI","RASI","RPSI","LPSI"])
    modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                  segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                  angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                  eulerSequences=["TOR","TOR", "ROT"],
                                  globalFrameOrientation = globalFrame,
                                  forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")   
                        
                        
    btkTools.smartWriter(acqGait, "fitting-cgm2_1.c3d")                        
                        

    # ------- OPENSIM IK --------------------------------------

    # --- osim builder ---
    cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model)
    markersetFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml"

    osimfile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    


    oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                            model,
                                            cgmCalibrationprocedure)
    oscf.addMarkerSet(markersetFile)
    scalingOsim = oscf.build()
    oscf.exportXml("OSIMTEST.osim")
    
    
    # --- fitting ---    
    
    #procedure 
    
    cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model)
#    cgmFittingProcedure.updateMarkerWeight("LASI",100)
#    cgmFittingProcedure.updateMarkerWeight("RASI",100)
#    cgmFittingProcedure.updateMarkerWeight("LPSI",100)
#    cgmFittingProcedure.updateMarkerWeight("RPSI",100)
#
#    cgmFittingProcedure.updateMarkerWeight("RTHI",100)
#    cgmFittingProcedure.updateMarkerWeight("RKNE",100)
#    cgmFittingProcedure.updateMarkerWeight("RTIB",100)
#    cgmFittingProcedure.updateMarkerWeight("RANK",100)
#    cgmFittingProcedure.updateMarkerWeight("RHEE",100)
#    cgmFittingProcedure.updateMarkerWeight("RTOE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTHI",100)
#    cgmFittingProcedure.updateMarkerWeight("LKNE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTIB",100)
#    cgmFittingProcedure.updateMarkerWeight("LANK",100)
#    cgmFittingProcedure.updateMarkerWeight("LHEE",100)
#    cgmFittingProcedure.updateMarkerWeight("LTOE",100)
    
       
    
    iksetupFile = pyCGM2.CONFIG.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml"
    
    osrf = opensimFilters.opensimFittingFilter(iksetupFile, 
                                                      scalingOsim, 
                                                      cgmFittingProcedure,
                                                      MAIN_PATH )
    acqIK = osrf.run(acqGait,str(MAIN_PATH + gaitFilename ))
    
    btkTools.smartWriter(acqIK,"fitting-cgm2_3.c3d")
    
    # -------- NEW MOTION FILTER ON IK MARKERS ------------------
    
    modMotion_ik=modelFilters.ModelMotionFilter(scp,acqIK,model,pyCGM2Enums.motionMethod.Sodervisk,
                                                useForMotionTest=True)
    modMotion_ik.compute()
    
    finalJcs =modelFilters.ModelJCSFilter(model,acqIK)
    finalJcs.setFilterBool(False) 
    finalJcs.compute(description="ik", pointLabelSuffix = "2_ik")#
    

    
    btkTools.smartWriter(acqIK,"fitting-cgm2_3-angles.c3d")
                        
                        
     

