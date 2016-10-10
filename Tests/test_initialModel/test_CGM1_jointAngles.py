# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 13:59:48 2016

@author: aaa34169


I prefer numpy.testing than unitest. 
easy to debug and assert method better suits. 

"""


import numpy as np
import scipy as sp

import pdb
import logging

try:
    import pyCGM2.pyCGM2_CONFIG 
except ImportError:
    logging.error("[pyCGM2] : pyCGM2 module not in your python path")



# pyCGM2
from pyCGM2.Core.Tools import  btkTools
from pyCGM2.Core.Model.CGM2 import cgm, modelFilters, modelDecorator, frame
import pyCGM2.Core.enums as pyCGM2Enums





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

class CGM1_motionJCSTest(): 

    @classmethod
    def basicCGM1(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def basicCGM1_flatFoot(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True).compute() 

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

    @classmethod
    def advancedCGM1_kad_noOptions(cls):  
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 
        


        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute(displayMarkers = True)
        
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, 
                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_kad", 
                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_kad").compute()

        
        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

    @classmethod
    def advancedCGM1_kad_flatFoot(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 
        


        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute(displayMarkers = True)
        
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                   leftFlatFoot = True, rightFlatFoot = True,
                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_kad", 
                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_kad").compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):      
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        # --- calibration                                    
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 
        
        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute(displayMarkers = True)
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")
        
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, 
                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_mid", 
                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_mid",
                                   useLeftTibialTorsion = True,useRightTibialTorsion = True).compute()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                        useLeftTibialTorsion = True,useRightTibialTorsion = True)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
#        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

#        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                        useLeftTibialTorsion = True,useRightTibialTorsion = True)
        modMotion.compute()

        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
#        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

#        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
#                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

class CGM1_motionAbsoluteAnglesTest():
    
    @classmethod
    def basicCGM1_absoluteAngles_lowerLimb(cls):  #def basicCGM1(self):    
        """
                
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")
        
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)                                        


        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")
        
        # ---   tests on angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)                                                                               


class CGM1_motionFullAnglesTest():
    @classmethod
    def basicCGM1(cls):     
        """
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()     
        markerDiameter=14                    
        mp={
        'mass'   : 71.0,                
        'leftLegLength' : 860.0,
        'rightLegLength' : 865.0 ,
        'leftKneeWidth' : 102.0,
        'rightKneeWidth' : 103.4,
        'leftAnkleWidth' : 75.3,
        'rightAnkleWidth' : 72.9,       
        }        
        model.addAnthropoInputParameter(mp)
                                    
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        # absolute angles 
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")        
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)


        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)                                                                               

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

         # relative angles
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="cgm1_6dof")
        
        # absolute angles 
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                      segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                      angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                      globalFrameOrientation = globalFrame,
                                      forwardProgression = forwardProgression).compute(pointLabelSuffix="cgm1_6dof")  
        
        #btkTools.smartWriter(acqGait, "test.c3d")   
   
        # tests on joint angles
        np.testing.assert_almost_equal( acqGait.GetPoint("RHipAngles").GetValues(),
                                        acqGait.GetPoint("RHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LHipAngles").GetValues(),
                                        acqGait.GetPoint("LHipAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("RKneeAngles").GetValues(),
                                        acqGait.GetPoint("RKneeAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LKneeAngles").GetValues(),
                                        acqGait.GetPoint("LKneeAngles_cgm1_6dof").GetValues(), decimal =3)   
   
        np.testing.assert_almost_equal( acqGait.GetPoint("RAnkleAngles").GetValues(),
                                        acqGait.GetPoint("RAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        np.testing.assert_almost_equal( acqGait.GetPoint("LAnkleAngles").GetValues(),
                                        acqGait.GetPoint("LAnkleAngles_cgm1_6dof").GetValues(), decimal =3)

        # tests on absolute angles
        np.testing.assert_almost_equal( acqGait.GetPoint("LFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("LFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RFootProgressAngles").GetValues(),
                                        acqGait.GetPoint("RFootProgressAngles_cgm1_6dof").GetValues(), decimal =3)
        np.testing.assert_almost_equal( acqGait.GetPoint("RPelvisAngles").GetValues(),
                                        acqGait.GetPoint("RPelvisAngles_cgm1_6dof").GetValues(), decimal =3) 
        np.testing.assert_almost_equal( acqGait.GetPoint("LPelvisAngles").GetValues(),
                                        acqGait.GetPoint("LPelvisAngles_cgm1_6dof").GetValues(), decimal =3)                                              
if __name__ == "__main__":

    CGM1_motionJCSTest.basicCGM1()
    CGM1_motionJCSTest.basicCGM1_flatFoot()  
    CGM1_motionJCSTest.advancedCGM1_kad_noOptions()
    CGM1_motionJCSTest.advancedCGM1_kad_flatFoot()
    CGM1_motionJCSTest.advancedCGM1_kad_midMaleolus()
    
    CGM1_motionAbsoluteAnglesTest.basicCGM1_absoluteAngles_lowerLimb()
    
    CGM1_motionFullAnglesTest.basicCGM1()