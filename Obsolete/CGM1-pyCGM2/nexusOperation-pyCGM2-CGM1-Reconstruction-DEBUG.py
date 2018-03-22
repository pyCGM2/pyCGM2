# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:41:34 2016

@author: Fabien Leboeuf
"""
import os
import logging
import matplotlib.pyplot as plt 
import json
import pdb


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus

# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body

#btk
pyCGM2.CONFIG.addBtk()  
import btk



# pyCGM2 libraries
from pyCGM2.Model.CGM2 import cgm, modelFilters,forceplates,bodySegmentParameters
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2 import  smartFunctions 

    
if __name__ == "__main__":
    
    plt.close("all")
    #pyNEXUS = ViconNexus.ViconNexus()    
    #NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    NEXUS_PYTHON_CONNECTED = True   
     
    if NEXUS_PYTHON_CONNECTED: # run Operation

        #---- INPUTS------
        Calibration = False
        if Calibration:
            calibrateFilenameLabelledNoExt = None   
        else:
            calibrateFilenameLabelledNoExt = "static Cal 01"
     
        flag_leftFlatFoot =  True
        flag_rightFlatFoot =  True
        markerDiameter = 14
        normativeDataInput = "Schwartz2008_Free"
        pointSuffix = "tr"
        enableProcessing = True
    
        #---- DATA ------ 
        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\pyCGM2-CGM1-basic\\"


        if calibrateFilenameLabelledNoExt is None:
            logging.warning("Static Processing")
            staticProcessing = True
            calibrateFilenameLabelled = "static Cal 01" + ".c3d" 
            reconstructFilenameLabelledNoExt = "static Cal 01"  
            reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        else:
            staticProcessing = False
            reconstructFilenameLabelledNoExt = "gait Trial 01"  
            reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
            calibrateFilenameLabelled = reconstructFilenameLabelled

        
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled ) 
        
        # subject mp
        mp={
        'Bodymass'   : 71.0,                
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,       
        }
        
 
        # -----------CGM STATIC CALIBRATION--------------------
        model=cgm.CGM1LowerLimbs()
        model.configure()
        model.addAnthropoInputParameter(mp)


        # reader
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # initial static filter
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter).compute() 


        # -----------CGM RECONSTRUCTION--------------------
        acqGait = btkTools.smartReader(str(DATA_PATH + reconstructFilenameLabelled))
 

        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter)

        modMotion.compute() 


        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)        

        if not staticProcessing:
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
                                 ).compute(pointLabelSuffix=pointSuffix)
                                 
    
            modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix=pointSuffix)
       
        # add metadata   
        md_Model = btk.btkMetaData('MODEL') # create main metadata
        btk.btkMetaDataCreateChild(md_Model, "NAME", "CGM1")
        btk.btkMetaDataCreateChild(md_Model, "PROCESSOR", "pyCGM2")
        acqGait.GetMetaData().AppendChild(md_Model)
       
       
        # writer
        btkTools.smartWriter(acqGait,str(DATA_PATH + reconstructFilenameLabelled[:-4] + "_cgm1.c3d"))
        logging.info( "[pyCGM2] : file ( %s) reconstructed in pyCGM2-model path " % (reconstructFilenameLabelled))


        # -----------CGM PROCESSING--------------------
        if enableProcessing:
            # infos        
            model= None 
            subject=None       
            experimental=None
    
            if staticProcessing:
                # temporal static angle and static angle profile
                smartFunctions.staticProcessing_cgm1(str(reconstructFilenameLabelled[:-4] + "_cgm1.c3d"), DATA_PATH,
                                                     model,  subject, experimental,
                                                     pointLabelSuffix = pointSuffix)            
            else:
     
                normativeData = { "Author": normativeDataInput[:normativeDataInput.find("_")],"Modality": normativeDataInput[normativeDataInput.find("_")+1:]} 
                             
                # ----PROCESSING-----
                smartFunctions.gaitProcessing_cgm1 (str(reconstructFilenameLabelled[:-4] + "_cgm1.c3d"), DATA_PATH,
                                       model,  subject, experimental,
                                       pointLabelSuffix = pointSuffix,
                                       plotFlag= True, 
                                       exportBasicSpreadSheetFlag = False,
                                       exportAdvancedSpreadSheetFlag = False,
                                       exportAnalysisC3dFlag = False,
                                       consistencyOnly = True,
                                       normativeDataDict = normativeData)
   
    else: 
        logging.error("Nexus Not Connected")     
         
  
    