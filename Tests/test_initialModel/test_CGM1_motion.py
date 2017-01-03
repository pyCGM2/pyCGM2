# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 13:59:48 2016

@author: aaa34169


I prefer numpy.testing than unitest. 
easy to debug and assert method better suits. 

"""


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

class CGM1_motionTest(): 

    @classmethod
    def basicCGM1(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf() 
        model.configure()
        
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
                                    
        # CALIBRATION
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute() 

        # --- Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        


        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)
        


        # --- Test 2 Motion Axe -X ----
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)       
        
       


    @classmethod
    def basicCGM1_flatFoot(cls):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic FlatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()
        model.configure()
        
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, leftFlatFoot = True, rightFlatFoot = True).compute() 
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        # ------ Test 1 Motion Axe X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 14.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)


        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)
        

    @classmethod
    def advancedCGM1_kad_noOptions(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()
        model.configure()
        
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

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)
                                        
        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)

    @classmethod
    def advancedCGM1_kad_flatFoot(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-flatFoot\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()
        model.configure()        
        
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

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)

        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation() 
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        np.testing.assert_almost_equal( R_leftShank,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShank,
                                        R_rightShankVicon, decimal =3)

        #       foot
        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
        np.testing.assert_almost_equal( R_leftFoot,
                                        R_leftFootVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightFoot,
                                        R_rightFootVicon, decimal =3)
      
        
    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):      
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d" 
    
        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))    
        
        model=cgm.CGM1ModelInf()
        model.configure()
        
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

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        
        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


#        #foot (Do not consider since Vicon Foot employs wrong shank axis)
#        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
#        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
#        np.testing.assert_almost_equal( R_leftFoot,
#                                        R_leftFootVicon, decimal =3)
#        np.testing.assert_almost_equal( R_rightFoot,
#                                        R_rightFootVicon, decimal =3)



        # ------ Test 2 Motion Axe -X -------
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 12.c3d"        
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))

        
        # Motion FILTER 
        # optimisation segmentaire et calibration fonctionnel
        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                        useLeftTibialTorsion = True,useRightTibialTorsion = True)
        modMotion.compute()

        #btkTools.smartWriter(acqGait, "test.c3d")        
        
        # ---     tests joint centre trajectory
        # joint centres trajectory
        np.testing.assert_almost_equal( acqGait.GetPoint("LFEP").GetValues(),
                                        acqGait.GetPoint("LHJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEP").GetValues(),
                                        acqGait.GetPoint("RHJC").GetValues(), decimal =3)


        np.testing.assert_almost_equal( acqGait.GetPoint("LFEO").GetValues(),
                                        acqGait.GetPoint("LKJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RFEO").GetValues(),
                                        acqGait.GetPoint("RKJC").GetValues(), decimal =3)
        
        np.testing.assert_almost_equal( acqGait.GetPoint("LTIO").GetValues(),
                                        acqGait.GetPoint("LAJC").GetValues(), decimal =3)        
        
        np.testing.assert_almost_equal( acqGait.GetPoint("RTIO").GetValues(),
                                        acqGait.GetPoint("RAJC").GetValues(), decimal =3) 

        # ---     tests on anatomical referential
        R_leftThigh= model.getSegment("Left Thigh").anatomicalFrame.motion[10].getRotation()
        R_leftShank= model.getSegment("Left Shank").anatomicalFrame.motion[10].getRotation()
        R_leftShankProx= model.getSegment("Left Shank Proximal").anatomicalFrame.motion[10].getRotation()
        R_leftFoot= model.getSegment("Left Foot").anatomicalFrame.motion[10].getRotation()
        
        R_rightThigh= model.getSegment("Right Thigh").anatomicalFrame.motion[10].getRotation()
        R_rightShank= model.getSegment("Right Shank").anatomicalFrame.motion[10].getRotation()
        R_rightShankProx= model.getSegment("Right Shank Proximal").anatomicalFrame.motion[10].getRotation()

        R_rightFoot= model.getSegment("Right Foot").anatomicalFrame.motion[10].getRotation()

        #       thigh
        R_leftThighVicon = getViconRmatrix(10, acqGait, "LFEO", "LFEP", "LFEL", "ZXiY")
        R_rightThighVicon = getViconRmatrix(10, acqGait, "RFEO", "RFEP", "RFEL", "ZXiY")

        np.testing.assert_almost_equal( R_leftThigh,
                                        R_leftThighVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightThigh,
                                        R_rightThighVicon, decimal =3)

        #       shank
        R_leftShankVicon = getViconRmatrix(10, acqGait, "LTIO", "LTIP", "LTIL", "ZXiY") 
        R_rightShankVicon = getViconRmatrix(10, acqGait, "RTIO", "RTIP", "RTIL", "ZXiY")
        
        np.testing.assert_almost_equal( R_leftShankProx,
                                        R_leftShankVicon, decimal =3)
        np.testing.assert_almost_equal( R_rightShankProx,
                                        R_rightShankVicon, decimal =3)


#        #foot (Do not consider since Vicon Foot employs wrong shank axis)
#        R_leftFootVicon = getViconRmatrix(10, acqGait, "LFOO", "LFOP", "LFOL", "ZXiY") 
#        R_rightFootVicon = getViconRmatrix(10, acqGait, "RFOO", "RFOP", "RFOL", "ZXiY") 
#        np.testing.assert_almost_equal( R_leftFoot,
#                                        R_leftFootVicon, decimal =3)
#        np.testing.assert_almost_equal( R_rightFoot,
#                                        R_rightFootVicon, decimal =3)



if __name__ == "__main__":

    logging.info("######## PROCESS CGM1 ######")
    CGM1_motionTest.basicCGM1()
    CGM1_motionTest.basicCGM1_flatFoot()
    CGM1_motionTest.advancedCGM1_kad_noOptions()
    CGM1_motionTest.advancedCGM1_kad_flatFoot()
    CGM1_motionTest.advancedCGM1_kad_midMaleolus()
    logging.info("######## PROCESS CGM1 --> Done######")
