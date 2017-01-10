# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:42:19 2016

@author: Fabien Leboeuf (Salford Univ)


# FUNCTIONS IN PROGRESS.


"""

# -- classic packages --    
import logging
import pdb
import os


# pyCGM package
import pyCGM2.Processing.cycle as CGM2cycle
import pyCGM2.Tools.trialTools as CGM2trialTools
import pyCGM2.Processing.analysis as CGM2analysis
import pyCGM2.Report.plot as CGM2plot
import pyCGM2.Report.normativeDatabaseProcedure as CGM2normdata
from pyCGM2.Processing import analysis


# openma
import ma.io


def staticProcessing_cgm1(modelledStaticFilename, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo,
                         pointLabelSuffix = "",
                         name_out=None,  DATA_PATH_OUT= None ):

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix   

    # reader
    kinematicFileNode = ma.io.read(str(DATA_PATH + modelledStaticFilename))
    kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)


    # --- common temporal plot
    temporelPlotPdf = CGM2plot.gaitKinematicsTemporalPlotPanel(kinematicTrial,pointLabelSuffix=pointLabelSuffix,  filename=modelledStaticFilename, path = DATA_PATH)

    
    # --- static angle profile
    # parameters
    angles =["LHipAngles"+pointLabelSuffixPlus,"LKneeAngles"+pointLabelSuffixPlus,"LAnkleAngles"+pointLabelSuffixPlus,"LFootProgressAngles"+pointLabelSuffixPlus,"LPelvisAngles"+pointLabelSuffixPlus,
                          "RHipAngles"+pointLabelSuffixPlus,"RKneeAngles"+pointLabelSuffixPlus,"RAnkleAngles"+pointLabelSuffixPlus,"RFootProgressAngles"+pointLabelSuffixPlus,"RPelvisAngles"+pointLabelSuffixPlus]

    # analysis
    staticAnalysis = analysis.staticAnalysisFilter(kinematicTrial,angles,
                            subjectInfos=subjectInfo,
                            modelInfos= modelInfo,
                            experimentalInfos=experimentalInfo)
    staticAnalysis.buildDataFrame()
    
    # plot
    if DATA_PATH_OUT is None:
        DATA_PATH_OUT = DATA_PATH

    plotBuilder = CGM2plot.StaticAnalysisPlotBuilder(staticAnalysis.m_dataframe,pointLabelSuffix=pointLabelSuffix, staticModelledFilename = modelledStaticFilename)
    # Filter
    pf = CGM2plot.PlottingFilter()
    pf.setBuilder(plotBuilder)
    pf.setPath(DATA_PATH_OUT)
    if name_out  is None:
        pdfname = modelledStaticFilename[:-4] 
    else:
        pdfname = name_out
    
    pf.setPdfName(pdfname)
    pf.plot()

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
        CGM2trialTools.sortedEvents(kinematicTrial)

        if longitudinal_axis is None or lateral_axis is None:
            logging.info("Automatic detection of Both longitudinal and lateral Axes")
            longitudinalAxis,forwardProgression,globalFrame = CGM2trialTools.findProgressionFromPoints(kinematicTrial,"LPSI","LASI","RPSI")
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
    kineticTrials,kineticFilenames,flag_kinetics =  CGM2trialTools.automaticKineticDetection(DATA_PATH,modelledFilenames)                         

    #---- GAIT CYCLES FILTER
    #--------------------------------------------------------------------------
    cycleBuilder = CGM2cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None,
                                               longitudinal_axis= globalFrame[0],lateral_axis=globalFrame[1])
            
    cyclefilter = CGM2cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #---- GAIT ANALYSIS FILTER
    #--------------------------------------------------------------------------

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

    kinematicLabelsDict ={ 'Left': ["LHipAngles"+pointLabelSuffixPlus,"LKneeAngles"+pointLabelSuffixPlus,"LAnkleAngles"+pointLabelSuffixPlus,"LFootProgressAngles"+pointLabelSuffixPlus,"LPelvisAngles"+pointLabelSuffixPlus],
                    'Right': ["RHipAngles"+pointLabelSuffixPlus,"RKneeAngles"+pointLabelSuffixPlus,"RAnkleAngles"+pointLabelSuffixPlus,"RFootProgressAngles"+pointLabelSuffixPlus,"RPelvisAngles"+pointLabelSuffixPlus] }

    kineticLabelsDict ={ 'Left': ["LHipMoment"+pointLabelSuffixPlus,"LKneeMoment"+pointLabelSuffixPlus,"LAnkleMoment"+pointLabelSuffixPlus, "LHipPower"+pointLabelSuffixPlus,"LKneePower"+pointLabelSuffixPlus,"LAnklePower"+pointLabelSuffixPlus],
                    'Right': ["RHipMoment"+pointLabelSuffixPlus,"RKneeMoment"+pointLabelSuffixPlus,"RAnkleMoment"+pointLabelSuffixPlus, "RHipPower"+pointLabelSuffixPlus,"RKneePower"+pointLabelSuffixPlus,"RAnklePower"+pointLabelSuffixPlus]}


    analysisBuilder = CGM2analysis.GaitAnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  subjectInfos=subjectInfo,
                                                  modelInfos=modelInfo,
                                                  experimentalInfos=experimentalInfo)
    
    analysisFilter = CGM2analysis.AnalysisFilter()
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
        plotBuilder = CGM2plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=flag_kinetics, pointLabelSuffix= pointLabelSuffix)
        if normativeDataDict["Author"] == "Schwartz2008":
            chosenModality = normativeDataDict["Modality"]
            plotBuilder.setNormativeDataProcedure(CGM2normdata.Schwartz2008_normativeDataBases(chosenModality)) # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
        elif normativeDataDict["Author"] == "Pinzone2014":
            chosenModality = normativeDataDict["Modality"]
            plotBuilder.setNormativeDataProcedure(CGM2normdata.Pinzone2014_normativeDataBases(chosenModality)) # modalites : "Center One" ,"Center Two"
       
        plotBuilder.setConsistencyOnly(consistencyOnly)       
       
        # Filter
        pf = CGM2plot.PlottingFilter()
        pf.setBuilder(plotBuilder)
        pf.setPath(DATA_PATH_OUT)
 
        if name_out  is None:
            pdfName = modelledFilenames[0][:-4] if len(modelledFilenames) == 1 else  "MultiTrials"
        else:
            pdfName = name_out
        
        pf.setPdfName(pdfName)
        pf.plot()

        os.startfile(DATA_PATH+"consistencyKinematics_"+ pdfName +".pdf")
        if flag_kinetics: os.startfile(DATA_PATH+"consistencyKinetics_"+ pdfName+".pdf")


        