# -*- coding: utf-8 -*-

# -- classic packages --    
import logging
import pdb
import os


from pyCGM2.Processing import cycle,analysis
from pyCGM2.Report import plot,normativeDatabaseProcedure
from pyCGM2.Tools import trialTools

# openma
import ma.io


def staticProcessing_cgm1(modelledStaticFilename, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo,
                         exportSpreadSheet=False,
                         pointLabelSuffix = "",
                         name_out=None,  DATA_PATH_OUT= None ):
    """
        Process a static c3d with lower limb CGM outputs
        
        :Parameters:
           - `modelledStaticFilename` (str) - filename of the static c3d including cgm kinematics 
           - `DATA_PATH` (str) - folder ofthe static file ( must end with \\)    
           - `modelInfo` (dict) - info about the model    
           - `subjectInfo` (dict) -  info about the subject             
           - `experimentalInfo` (dict) - info about experimental conditions               
           - `exportSpreadSheet` (bool) - flag enable xls export    
           - `pointLabelSuffix` (str) - suffix added to standard cgm nomenclature    
           - `name_out` (str) - new filename of any output file ( instead modelledStaticFilename)     
           - `DATA_PATH_OUT` (str) - new folder to store any output file    
    """
    
    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix   

    # reader
    kinematicFileNode = ma.io.read(str(DATA_PATH + modelledStaticFilename))
    kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)

    # --- common temporal plot
    temporelPlotPdf = plot.gaitKinematicsTemporalPlotPanel(kinematicTrial,modelledStaticFilename, pointLabelSuffix=pointLabelSuffix, path = DATA_PATH)

    
    # --- static angle profile
    # parameters
    angles =[str("LHipAngles"+pointLabelSuffixPlus),str("LKneeAngles"+pointLabelSuffixPlus),str("LAnkleAngles"+pointLabelSuffixPlus),str("LFootProgressAngles"+pointLabelSuffixPlus),str("LPelvisAngles"+pointLabelSuffixPlus),
             str("RHipAngles"+pointLabelSuffixPlus),str("RKneeAngles"+pointLabelSuffixPlus),str("RAnkleAngles"+pointLabelSuffixPlus),str("RFootProgressAngles"+pointLabelSuffixPlus),str("RPelvisAngles"+pointLabelSuffixPlus)]

    # analysis
    staticAnalysis = analysis.StaticAnalysisFilter(kinematicTrial,angles,
                            subjectInfos=subjectInfo,
                            modelInfos= modelInfo,
                            experimentalInfos=experimentalInfo)
    staticAnalysis.build()
    
  
    
    # plot
    if DATA_PATH_OUT is None:
        DATA_PATH_OUT = DATA_PATH

    if exportSpreadSheet:
        if name_out  is None:
            spreadSheetName = modelledStaticFilename[:-4] 
        else:
            spreadSheetName = name_out

        staticAnalysis.exportDataFrame(spreadSheetName, path=DATA_PATH_OUT) 

    plotBuilder = plot.StaticAnalysisPlotBuilder(staticAnalysis.analysis,pointLabelSuffix=pointLabelSuffix, staticModelledFilename = modelledStaticFilename)
    # Filter
    pf = plot.PlottingFilter()
    pf.setBuilder(plotBuilder)
    pf.setPath(DATA_PATH_OUT)
    if name_out  is None:
        pdfname = modelledStaticFilename[:-4] 
    else:
        pdfname = name_out
    
    pf.setPdfName(pdfname)
    pf.plot()

    if name_out  is None:
        os.startfile(DATA_PATH+temporelPlotPdf)
        os.startfile(DATA_PATH+"staticAngleProfiles_"+ pdfname +".pdf")



def gaitProcessing_cgm1 (modelledFilenames, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo,
                         pointLabelSuffix = "",
                         plotFlag= True, 
                         exportBasicSpreadSheetFlag = True,
                         exportAdvancedSpreadSheetFlag = True,
                         exportAnalysisC3dFlag = True,
                         consistencyOnly = False,
                         normativeDataDict = None,
                         name_out=None,
                         DATA_PATH_OUT= None,
                         longitudinal_axis=None,
                         lateral_axis = None
                         ):
    """
        Processing  of multiple gait c3d with lower limb CGM outputs
        
        :Parameters:
           - `modelledFilenames` (list of str) - filenames of gai trial with kinematics and kinetics  
           - `DATA_PATH` (str) - folder where  modelledFilenames are stored  
           - `modelInfo` (dict) - info about the model   
           - `subjectInfo` (dict) - info about the subject               
           - `experimentalInfo` (dict) - info about experimental conditions               
           - `pointLabelSuffix` (str) - suffix added to standard cgm nomenclature    
           - `exportBasicSpreadSheetFlag` (bool) - enable xls export of a basic spreadsheet    
           - `exportAdvancedSpreadSheetFlag` (bool) - enable xls export of an advanced spreadsheet 
           - `exportAnalysisC3dFlag` (bool) - export a single 101-frames c3d storing all gait cycle  
           - `longitudinal_axis` (str) - label of the global longitudinal axis    
           - `lateral_axis` (str) - label of the global lateral axis
           - `name_out` (str) - new filename of any output files   
           - `DATA_PATH_OUT` (str) - new folder to store any  output files

   
       *TODO* :
       
           - better manage both longitudinal_axis and laterals_axis inputs. 

    """


                             
    #---- PRELIMINARY STAGE
    #--------------------------------------------------------------------------
    # check if modelledFilenames is string                          
    if isinstance(modelledFilenames,str) or isinstance(modelledFilenames,unicode):
        logging.info( "gait Processing on ONE file")        
        modelledFilenames = [modelledFilenames]
    


    # distinguishing trials for kinematic and kinetic processing                             
    # - kinematic Trials      
    kinematicTrials=[]
    kinematicFilenames =[]
    for kinematicFilename in modelledFilenames:
        kinematicFileNode = ma.io.read(str(DATA_PATH + kinematicFilename))
        kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
        trialTools.sortedEvents(kinematicTrial)

        if longitudinal_axis is None or lateral_axis is None:
            logging.info("Automatic detection of Both longitudinal and lateral Axes")
            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgressionFromPoints(kinematicTrial,"LPSI","LASI","RPSI")
        else:    
            if longitudinal_axis is None or lateral_axis is not None:
                raise Exception("[pyCGM2] Longitudinal_axis has to be also defined")     
            if longitudinal_axis is not None or lateral_axis is None:
                raise Exception("[pyCGM2] Lateral_axis has to be also defined")     
    
            if longitudinal_axis is not None or lateral_axis is not None:
                globalFrame[0] = longitudinal_axis
                globalFrame[1] = lateral_axis

        kinematicTrials.append(kinematicTrial)
        kinematicFilenames.append(kinematicFilename)

    # - kinetic Trials ( check if kinetic events)        
    kineticTrials,kineticFilenames,flag_kinetics =  trialTools.automaticKineticDetection(DATA_PATH,modelledFilenames)                         

    #---- GAIT CYCLES FILTER
    #--------------------------------------------------------------------------
    cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None,
                                               longitudinal_axis= globalFrame[0],lateral_axis=globalFrame[1])
            
    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #---- GAIT ANALYSIS FILTER
    #--------------------------------------------------------------------------

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

    kinematicLabelsDict ={ 'Left': [str("LHipAngles"+pointLabelSuffixPlus),str("LKneeAngles"+pointLabelSuffixPlus),str("LAnkleAngles"+pointLabelSuffixPlus),str("LFootProgressAngles"+pointLabelSuffixPlus),str("LPelvisAngles"+pointLabelSuffixPlus)],
                           'Right': [str("RHipAngles"+pointLabelSuffixPlus),str("RKneeAngles"+pointLabelSuffixPlus),str("RAnkleAngles"+pointLabelSuffixPlus),str("RFootProgressAngles"+pointLabelSuffixPlus),str("RPelvisAngles"+pointLabelSuffixPlus)] }

    kineticLabelsDict ={ 'Left': [str("LHipMoment"+pointLabelSuffixPlus),str("LKneeMoment"+pointLabelSuffixPlus),str("LAnkleMoment"+pointLabelSuffixPlus), str("LHipPower"+pointLabelSuffixPlus),str("LKneePower"+pointLabelSuffixPlus),str("LAnklePower"+pointLabelSuffixPlus)],
                    'Right': [str("RHipMoment"+pointLabelSuffixPlus),str("RKneeMoment"+pointLabelSuffixPlus),str("RAnkleMoment"+pointLabelSuffixPlus), str("RHipPower"+pointLabelSuffixPlus),str("RKneePower"+pointLabelSuffixPlus),str("RAnklePower"+pointLabelSuffixPlus)]}


    analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  subjectInfos=subjectInfo,
                                                  modelInfos=modelInfo,
                                                  experimentalInfos=experimentalInfo)
    
    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()
    
    
    # export dataframe
    if DATA_PATH_OUT is None:
        DATA_PATH_OUT = DATA_PATH
        
    if exportAnalysisC3dFlag:
        if name_out  is None:
            c3dAnalysisName = modelledFilenames[0][:-4]+"-Cycles" if len(modelledFilenames) == 1 else  "MultiTrials"
        else:
            c3dAnalysisName = name_out
            
        analysisFilter.exportAnalysisC3d(c3dAnalysisName, path=DATA_PATH_OUT)

    if exportBasicSpreadSheetFlag or exportAdvancedSpreadSheetFlag:
        if name_out  is None:
            spreadSheetName = modelledFilenames[0][:-4] if len(modelledFilenames) == 1 else  "MultiTrials"
        else:
            spreadSheetName = name_out

        if exportBasicSpreadSheetFlag : analysisFilter.exportBasicDataFrame(spreadSheetName, path=DATA_PATH_OUT)
        if exportAdvancedSpreadSheetFlag : analysisFilter.exportAdvancedDataFrame(spreadSheetName, path=DATA_PATH_OUT)

    #---- GAIT PLOTTING FILTER
    #--------------------------------------------------------------------------
    if plotFlag:    
        
        plotBuilder = plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=flag_kinetics, pointLabelSuffix= pointLabelSuffix)
        
        if normativeDataDict["Author"] == "Schwartz2008":
            chosenModality = normativeDataDict["Modality"]
            plotBuilder.setNormativeDataProcedure(normativeDatabaseProcedure.Schwartz2008_normativeDataBases(chosenModality)) # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
        elif normativeDataDict["Author"] == "Pinzone2014":
            chosenModality = normativeDataDict["Modality"]
            plotBuilder.setNormativeDataProcedure(normativeDatabaseProcedure.Pinzone2014_normativeDataBases(chosenModality)) # modalites : "Center One" ,"Center Two"
       
      
        plotBuilder.setConsistencyOnly(consistencyOnly)       
       
        # Filter
        pf = plot.PlottingFilter()
        pf.setBuilder(plotBuilder)
        pf.setPath(DATA_PATH_OUT)
 
        if name_out  is None:
            pdfName = modelledFilenames[0][:-4] if len(modelledFilenames) == 1 else  "MultiTrials"
        else:
            pdfName = name_out
        
        pf.setPdfName(pdfName)
        pf.plot()

        if name_out  is None:
            os.startfile(DATA_PATH+"consistencyKinematics_"+ pdfName +".pdf")
            if flag_kinetics: os.startfile(DATA_PATH+"consistencyKinetics_"+ pdfName+".pdf")


        