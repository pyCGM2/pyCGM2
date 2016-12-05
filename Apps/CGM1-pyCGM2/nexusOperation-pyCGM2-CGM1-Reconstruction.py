# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:41:34 2016

@author: Fabien Leboeuf
"""
import os
import logging
import matplotlib.pyplot as plt 
import json
import sys

# pyCGM2 settings
import pyCGM2
pyCGM2.pyCGM2_CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.pyCGM2_CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.pyCGM2_CONFIG.addOpenma()
import ma.io
import ma.body

# pyCGM2 libraries
from pyCGM2.Core.Model.CGM2 import cgm, modelFilters, modelDecorator,forceplates,bodySegmentParameters
from pyCGM2.Core.Tools import btkTools
import pyCGM2.Core.enums as pyCGM2Enums
import pyCGM2.smartFunctions as CGM2smart 

    
if __name__ == "__main__":
    
    plt.close("all")
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    #NEXUS_PYTHON_CONNECTED = True   
     
    if NEXUS_PYTHON_CONNECTED: # run Operation


        #---- INPUTS------       
        calibrateFilenameLabelled = sys.argv[1] 
        flag_leftFlatFoot =  bool(int(sys.argv[2]))
        flag_rightFlatFoot =  bool(int(sys.argv[3]))
        markerDiameter =  float(sys.argv[4])        
        
        #---- DATA ----
        DATA_PATH, reconstructFilenameLabelledNoExt = pyNEXUS.GetTrialName()
        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled ) 
        
        # subject mp
        subjects = pyNEXUS.GetSubjectNames()
        subject =   subjects[0]   
        logging.info(  "Subject name : " + subject  )

        Parameters = pyNEXUS.GetSubjectParamNames(subject)
        
        mp={
        'mass'   : pyNEXUS.GetSubjectParamDetails( subject, "Bodymass")[0],                
        'leftLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0],
        'rightLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "rightLegLength")[0] ,
        'leftKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "leftKneeWidth")[0],
        'rightKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "rightKneeWidth")[0],
        'leftAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "leftAnkleWidth")[0],
        'rightAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "rightAnkleWidth")[0],       
        }
        
 
        # -----------CGM STATIC CALIBRATION--------------------
        model=cgm.CGM1ModelInf()
        model.configure()
        model.addAnthropoInputParameter(mp)


        # reader
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # initial static filter
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter).compute() 

        
#        # cgm decorator
#        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=25, displayMarkers = True)
#        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=25, side="both")
#        
#        # final static filter
#        modelFilters.ModelCalibrationFilter(scp,acqStatic,model, 
#                                   useLeftKJCnode="LKJC_kad", useLeftAJCnode="LAJC_mid", 
#                                   useRightKJCnode="RKJC_kad", useRightAJCnode="RAJC_mid",
#                                   useLeftTibialTorsion = True,useRightTibialTorsion = True,
#                                   markerDiameter=25).compute()

        # -----------CGM RECONSTRUCTION--------------------
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter)

        modMotion.compute() 


        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix="")

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix="")        

         # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        # force plate -- construction du wrench attribue au pied       
        forceplates.appendForcePlateCornerAsMarker(acqGait)       
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot", 
                                 rightSegmentLabel="Right Foot").compute()

        # Joint kinetics        
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             acqGait,
                             procedure = idp,
                             projection = pyCGM2Enums.MomentProjection.Distal
                             ).compute(pointLabelSuffix="")
                             

        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix="")
       
        # writer
        btkTools.smartWriter(acqGait,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_pyCGM2_CGM1.c3d"))
        logging.info( "[pyCGM2] : file ( %s) reconstructed in pyCGM2-model path " % (reconstructFilenameLabelled))



        # -----------CGM PROCESSING--------------------
    
        # inputs
        normativeDataInput = "Schwartz2008_VeryFast"
        normativeData = { "Author": normativeDataInput[:normativeDataInput.find("_")],"Modality": normativeDataInput[normativeDataInput.find("_")+1:]} 
    
        # infos        
        model= None 
        subject=None       
        experimental=None
                     
        # ----PROCESSING-----
        CGM2smart.gaitProcessing_cgm1 (str(reconstructFilenameLabelled[:-4] + "_pyCGM2_CGM1.c3d"), DATA_PATH,
                               model,  subject, experimental, 
                               plotFlag= True, 
                               exportBasicSpreadSheetFlag = False,
                               exportAdvancedSpreadSheetFlag = False,
                               exportAnalysisC3dFlag = False,
                               consistencyOnly = True,
                               normativeDataDict = normativeData)
   
    else: 
        logging.error("Nexus Not Connected")     
         
  
    