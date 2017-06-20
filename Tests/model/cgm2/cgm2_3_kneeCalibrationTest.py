# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
import pdb
import logging
import cPickle

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.DEBUG)

import pyCGM2
# btk
pyCGM2.CONFIG.addBtk() 
import btk 
    
# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm2, modelFilters, modelDecorator
import pyCGM2.enums as pyCGM2Enums

from pyCGM2.Model.Opensim import opensimFilters
import json
from collections import OrderedDict



# enableLongitudinalRotation in Static and Motion filter rotate along Z 
class CGM2_SARA_test(): 

    @classmethod
    def CGM2_3_SARA_test(cls):
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\Knee Calibration\\"
        staticFilename = "Static.c3d"
        
        leftKneeFilename = "Left Knee.c3d"
        rightKneeFilename = "Right Knee.c3d"
        gaitFilename= "gait trial 01.c3d"
        
        
        markerDiameter=14     
        mp={
        'Bodymass'   : 71,                
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,     
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
                "RTIB":"",
                "RANK":"",
                "RTIBAP":"RTIAP",
                "RTIBAD":"RTIAD",
                "RHEE":"",
                "RTOE":"",
                "LTHI":"",
                "LKNE":"",
                "LTHIAP":"LTHAP",
                "LTHIAD":"LTHAD",
                "LTIB":"",
                "LANK":"",
                "LTIBAP":"LTIAP",
                "LTIBAD":"LTIAD",
                "LHEE":"",
                "LTOE":""
                }
            }
          """        
        
        
        # --- Calibration ---                          
        model=cgm2.CGM2_3LowerLimbs()
        model.configure()        

        
        inputs = json.loads(CONTENT_INPUTS_CGM2_3,object_pairs_hook=OrderedDict)
        translators = inputs["Translators"]

#        f = open(MAIN_PATH +  'MRI-US-01-CGM2_3-pyCGM2.model', 'r')
#        model = cPickle.load(f)
#        f.close() 

        
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename),translators=translators)   
          
        model.addAnthropoInputParameters(mp)
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 
        
        # cgm decorator
        modelDecorator.HipJointCenterDecorator(model).hara()  
            
        # final
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useLeftHJCnode="LHJC_Hara", useRightHJCnode="RHJC_Hara").compute()

               
        
        # ------ LEFT KNEE CALIBRATION -------      
        acqLeftKnee = btkTools.smartReader(str(MAIN_PATH +  leftKneeFilename),translators=translators)
        
        # Motion of only left 
        modMotionLeftKnee=modelFilters.ModelMotionFilter(scp,acqLeftKnee,model,pyCGM2Enums.motionMethod.Sodervisk)
        modMotionLeftKnee.segmentalCompute(["Left Thigh","Left Shank"])    
        
        # decorator
        modelDecorator.KneeCalibrationDecorator(model).sara("Left")


        Or_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        axis_inThigh = model.getSegment("Left Thigh").getReferential("TF").getNodeTrajectory("KneeFlexionAxis")
        
        
        btkTools.smartAppendPoint(acqLeftKnee,"Left" +"_KneeFlexionOri",Or_inThigh)
        btkTools.smartAppendPoint(acqLeftKnee,"Left" +"_KneeFlexionAxis",axis_inThigh) 
       

        btkTools.smartWriter(acqLeftKnee, "CGM2_3_SARAfunc_test.c3d")
        
        # new static calibration          
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, 
                                            useLeftKJCnode="KJC_Sara",
                                            enableLongitudinalRotation=True
                                            ).compute()
        
        
#        # ------ Right KNEE CALIBRATION -------      
#        acqRightKnee = btkTools.smartReader(str(MAIN_PATH +  rightKneeFilename),translators=translators)
#        
#        # Motion FILTER 
#        modMotionRightKnee=modelFilters.ModelMotionFilter(scp,acqRightKnee,model,pyCGM2Enums.motionMethod.Sodervisk)
#        modMotionRightKnee.segmentalCompute(["Right Thigh","Right Shank"])    
#        
#        # cgm decorator
#        modelDecorator.KneeCalibrationDecorator(model).sara("Right")
#        
#        
#        # new static calibration with KJC node         
#        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, useRightKJCnode="KJC_sara").compute()
                            
        
        btkTools.smartWriter(acqStatic, "CGM2_3_SARA_test.c3d")
        

        # ------ Fitting -------      
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
    
        acqGait =  btkTools.applyTranslators(acqGait,translators)     
        # Motion FILTER 
    
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                                 enableLongitudinalRotation=True)
        modMotion.compute()
    
        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6nodof")
        

        btkTools.smartWriter(acqGait, "fitting-cgm2_3-withNoRottest.c3d")        

    

if __name__ == "__main__":

    CGM2_SARA_test.CGM2_3_SARA_test()


    

