# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums




if __name__ == "__main__":



    MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\dynaKAD\\"
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
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

    print model.m_useRightTibialTorsion

    # --- Test 1 Motion Axe X -------
    gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
    acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                             usePyCGM2_coordinateSystem=True)
    modMotion.compute()


    modelDecorator.KneeCalibrationDecorator(model).dynaKad("Left")
    modelDecorator.KneeCalibrationDecorator(model).dynaKad("Right")

    # calibration with dynaKad offset    
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

    # new motion with dynaKAD offset
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Determinist,
                                             usePyCGM2_coordinateSystem=True)
    modMotion.compute()

    btkTools.smartWriter(acqGait, "testDynaKAD.c3d")        


        
