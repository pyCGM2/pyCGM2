# -*- coding: utf-8 -*-

"""
 
Usage:
    file.py
    file.py -h | --help
    file.py --version
    file.py  Calibration [-lr] [--markerDiameter=<n>]  
    file.py  Calibration [-lr] [--markerDiameter=<n> --pointSuffix=<ps>] 
    file.py  <staticFile> [-lr] [--markerDiameter=<n>]
    file.py  <staticFile> [-lr] [--markerDiameter=<n>] [-p ]
    file.py  <staticFile> [-lr] [--markerDiameter=<n>] [-p  --author=<authorYear> --modality=<modalitfy>]
    file.py  <staticFile> [-lr] [--markerDiameter=<n> --pointSuffix=<ps>]         
    file.py  <staticFile> [-lr] [--markerDiameter=<n> --pointSuffix=<ps>] [-p | --plot --author=<authorYear> --modality=<modalitfy>]    
    
 
Arguments:

 
Options:
    -h --help   Show help message
    -l          Enable left flat foot option
    -r          Enable right flat foot option
    -p   Enable gait Plots  
    --markerDiameter=<n>  marker diameter [default: 14].
    --pointSuffix=<ps>  suffix associated with classic vicon output label  [default: ""].
    --author=<authorYear>   Name and year of the Normative Data base used [default: Schwartz2008]
    --modality=<modalitfy>  Modality of the Normative Database used  [default: Free]

"""

import os
import logging
import matplotlib.pyplot as plt 
import json
import sys
from docopt import docopt
import pdb
import btk

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
from pyCGM2.Core.Model.CGM2 import cgm, modelFilters,forceplates,bodySegmentParameters
from pyCGM2.Core.Tools import btkTools
import pyCGM2.Core.enums as pyCGM2Enums
from pyCGM2 import  smartFunctions 


    
if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')
        
    plt.close("all")
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    #NEXUS_PYTHON_CONNECTED = True   
     
    if NEXUS_PYTHON_CONNECTED: # run Operation


        #---- INPUTS------
        if args['Calibration']:
            calibrateFilenameLabelledNoExt = None  #sys.argv[1] 
        else:
            calibrateFilenameLabelledNoExt = args['<staticFile>']  #sys.argv[1] 

        flag_leftFlatFoot =  args['-l'] #bool(int(sys.argv[2]))
        flag_rightFlatFoot = args['-r'] #bool(int(sys.argv[3]))
        markerDiameter =  float(args['--markerDiameter']) #float(sys.argv[4])
        if  args['--pointSuffix'] == '""':
            pointSuffix = ""
        else:
            pointSuffix = args['--pointSuffix']   

        gaitProcessingEnable = args['-p']
        normativeDataInput = str(args['--author']+"_"+ args['--modality'])#"Schwartz2008_VeryFast"


        
        #---- DATA ----
        DATA_PATH, reconstructFilenameLabelledNoExt = pyNEXUS.GetTrialName()
        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        if calibrateFilenameLabelledNoExt is None:
            logging.warning("Static Processing")
            staticProcessing = True
            calibrateFilenameLabelled = reconstructFilenameLabelled
        else:
            staticProcessing = False
            calibrateFilenameLabelled = calibrateFilenameLabelledNoExt + ".c3d"

        
        logging.info( "data Path: "+ DATA_PATH )   
        logging.info( "calibration file: "+ calibrateFilenameLabelled)
        logging.info( "reconstruction file: "+ reconstructFilenameLabelled ) 
        
        # subject mp
                # subject mp
        subjects = pyNEXUS.GetSubjectNames()
        subject =   subjects[0]   
        logging.info(  "Subject name : " + subject  )

        Parameters = pyNEXUS.GetSubjectParamNames(subject)
        
        mp={
        'mass'   : pyNEXUS.GetSubjectParamDetails( subject, "Bodymass")[0],                
        'leftLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0],
        'rightLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0] ,
        'leftKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0],
        'rightKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0],
        'leftAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0],
        'rightAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0],       
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
        # infos        
        model= None 
        subject=None       
        experimental=None

        if staticProcessing:
            # static angle profile
            smartFunctions.staticProcessing_cgm1(str(reconstructFilenameLabelled[:-4] + "_cgm1.c3d"), DATA_PATH,
                                                 model,  subject, experimental,
                                                 pointLabelSuffix = pointSuffix)            
        else:
 
            if gaitProcessingEnable:
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
         
  
    