# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp

import pdb
import logging

try:
    import pyCGM2.pyCGM2_CONFIG
    pyCGM2.pyCGM2_CONFIG.setLoggingLevel(logging.DEBUG)
except ImportError:
    logging.error("[pyCGM2] : pyCGM2 module not in your python path")



# pyCGM2
from pyCGM2.Core.Tools import  btkTools
from pyCGM2.Core.Model.CGM2 import cgm, modelFilters,modelDecorator
import pyCGM2.Core.enums as pyCGM2Enums



class CGM1_calibrationTest(): 

    @classmethod
    def basicCGM1(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        
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


        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)
        
        

    @classmethod
    def basicCGM1_flatFoot(cls):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic FlatFoot\\"
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, leftFlatFoot = True, rightFlatFoot = True).compute() 
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        
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


        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)

    @classmethod
    def advancedCGM1_kad_noOptions(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-basic\\"
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

        
        # ---- Testing-------

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)
       
        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


        # foot offsets
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)
        
        
        # thigh and shank Offsets        
        lto = model.getViconThighOffset("Left")
        lso = model.getViconShankOffset("Left")
        rto = model.getViconThighOffset("Right")
        rso = model.getViconShankOffset("Right")

        
        np.testing.assert_almost_equal(lto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(lso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        

        np.testing.assert_almost_equal(rto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(rso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        


    @classmethod
    def advancedCGM1_kad_flatFoot(cls):  #def basicCGM1(self):    
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-flatFoot\\"
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

        
        # ---- Testing-------

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)
       
        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


        # foot offsets
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)
        
        
        # thigh and shank Offsets        
        lto = model.getViconThighOffset("Left")
        lso = model.getViconShankOffset("Left")
        rto = model.getViconThighOffset("Right")
        rso = model.getViconShankOffset("Right")

        
        np.testing.assert_almost_equal(lto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(lso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        

        np.testing.assert_almost_equal(rto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(rso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)               
      
        
    @classmethod
    def advancedCGM1_kad_midMaleolus(cls):      
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG advanced\\KAD-tibialTorsion\\"
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


         # ---- Testing-------

        # joint centres
        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEP").GetValues().mean(axis=0),acqStatic.GetPoint("LHJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEP").GetValues().mean(axis=0),acqStatic.GetPoint("RHJC").GetValues().mean(axis=0),decimal = 3)


        np.testing.assert_almost_equal(acqStatic.GetPoint("LFEO").GetValues().mean(axis=0),acqStatic.GetPoint("LKJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RFEO").GetValues().mean(axis=0),acqStatic.GetPoint("RKJC").GetValues().mean(axis=0),decimal = 3)
       
        np.testing.assert_almost_equal(acqStatic.GetPoint("LTIO").GetValues().mean(axis=0),acqStatic.GetPoint("LAJC").GetValues().mean(axis=0),decimal = 3)
        np.testing.assert_almost_equal(acqStatic.GetPoint("RTIO").GetValues().mean(axis=0),acqStatic.GetPoint("RAJC").GetValues().mean(axis=0),decimal = 3)


        # foot offsets
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        vicon_spf_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_spf_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticPlantFlex").value().GetInfo().ToDouble()[0])
        vicon_sro_l  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LStaticRotOff").value().GetInfo().ToDouble()[0])
        vicon_sro_r  = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RStaticRotOff").value().GetInfo().ToDouble()[0])


        np.testing.assert_almost_equal(spf_l,vicon_spf_l , decimal = 3)
        np.testing.assert_almost_equal(spf_r,vicon_spf_r , decimal = 3)
        np.testing.assert_almost_equal(sro_l,vicon_sro_l , decimal = 3)
        np.testing.assert_almost_equal(sro_r,vicon_sro_r , decimal = 3)
        
        
        # thigh and shank Offsets        
        lto = model.getViconThighOffset("Left")
        lso = model.getViconShankOffset("Left")
        rto = model.getViconThighOffset("Right")
        rso = model.getViconShankOffset("Right")

        
        np.testing.assert_almost_equal(lto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(lso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        

        np.testing.assert_almost_equal(rto, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RThighRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        
        np.testing.assert_almost_equal(rso, np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RShankRotation").value().GetInfo().ToDouble()[0]) , decimal = 3)        


        # shank abAdd offset
        np.testing.assert_almost_equal(-1.0 *model.mp_computed["leftAJCAbAdOffset"] , np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LAnkleAbAdd").value().GetInfo().ToDouble()[0]) , decimal = 3)
        np.testing.assert_almost_equal(model.mp_computed["rightAJCAbAdOffset"] , np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RAnkleAbAdd").value().GetInfo().ToDouble()[0]) , decimal = 3)

        btkTools.smartWriter(acqStatic, "outStatic_advancedCGM1_kad_midMaleolus.c3d") 


    


class CGM1_custom_calibrationTest(): 

    @classmethod
    def harrigton_fullPredictor(cls):      
        """
        GOAL : compare Joint centres and foot Offset
        
        """    
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
        
        
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
        MAIN_PATH = pyCGM2.pyCGM2_CONFIG.TEST_DATA_PATH + "CGM1\\PIG standard\\basic\\"
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
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            useBodyBuilderFoot=True).compute() 
        
        spf_l,sro_l,spf_r,sro_r = model.getViconFootOffset()

        
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
    CGM1_calibrationTest.advancedCGM1_kad_noOptions()
    CGM1_calibrationTest.advancedCGM1_kad_flatFoot()
    CGM1_calibrationTest.advancedCGM1_kad_midMaleolus()
    
      
    logging.info("######## PROCESS CGM1 --> Done ######")    
    
    # CGM1 - custom
    logging.info("######## PROCESS custom CGM1 ######")
    CGM1_custom_calibrationTest.harrigton_fullPredictor() 
    CGM1_custom_calibrationTest.harrigton_pelvisWidthPredictor()
    CGM1_custom_calibrationTest.harrigton_legLengthPredictor()  
    
    CGM1_custom_calibrationTest.customLocalPosition()
    CGM1_custom_calibrationTest.hara_regressions()
    CGM1_custom_calibrationTest.basicCGM1_BodyBuilderFoot()
    logging.info("######## PROCESS custom CGM1 --> Done ######")