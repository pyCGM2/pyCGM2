# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:42:19 2016

@author: Fabien Leboeuf (Salford Univ)


# FUNCTIONS IN PROGRESS. ( DO NOT USE) - need thinking more


"""

# -- classic packages --    


import ma.io

# pyCGM package
import pyCGM2.Core.Processing.cycle as CGM2cycle
import pyCGM2.Core.Tools.trialTools as CGM2trialTools
import pyCGM2.Core.Processing.analysis as CGM2analysis
import pyCGM2.Core.Report.plot as CGM2plot
import pyCGM2.Core.Report.normativeDatabaseProcedure as CGM2normdata







def gaitProcessing_cgm1 (modelledFilenames, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo, 
                         plotFlag= True, 
                         exportSpreadSheetFlag = True,
                         exportAnalysisC3dFlag = True,
                         normativeDataDict = None ):
                             
    #---- PRELIMINARY STAGE
    #--------------------------------------------------------------------------
    # check if modelledFilenames is string                          
    if isinstance(modelledFilenames,str) or isinstance(modelledFilenames,unicode):
        print " [pyCGM2] gait Processing on ONE file"        
        modelledFilenames = [modelledFilenames]
    


    # distinguishing trials for kinematic and kinetic processing                             
    # - kinematic Trials      
    kinematicTrials=[]
    kineticFilenames =[]
    for kinematicFilename in modelledFilenames:
        print kinematicFilename
        kinematicFileNode = ma.io.read(str(DATA_PATH + kinematicFilename))
        kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
        CGM2trialTools.sortedEvents(kinematicTrial)
        
        kinematicTrials.append(kinematicTrial)
        kineticFilenames.append(kinematicFilename)

    # - kinetic Trials ( check if kinetic events)        
    kineticTrials,kineticFilenames,flag_kinetics =  CGM2trialTools.automaticKineticDetection(DATA_PATH,modelledFilenames)                         


    #---- GAIT CYCLES FILTER
    #--------------------------------------------------------------------------
    cycleBuilder = CGM2cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None,
                                               longitudinal_axis=0,lateral_axis=1)
            
    cyclefilter = CGM2cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #---- GAIT ANALYSIS FILTER
    #--------------------------------------------------------------------------
    kinematicLabelsDict ={ 'Left': ["LHipAngles","LKneeAngles","LAnkleAngles","LFootProgressAngles","LPelvisAngles"],
                    'Right': ["RHipAngles","RKneeAngles","RAnkleAngles","RFootProgressAngles","RPelvisAngles"] }

    kineticLabelsDict ={ 'Left': ["LHipMoment","LKneeMoment","LAnkleMoment", "LHipPower","LKneePower","LAnklePower"],
                    'Right': ["RHipMoment","RKneeMoment","RAnkleMoment", "RHipPower","RKneePower","RAnklePower"]}


    analysisBuilder = CGM2analysis.GaitAnalysisBuilder(cycles,
                                                  kinematicLabelsDict = kinematicLabelsDict,
                                                  kineticLabelsDict = kineticLabelsDict,
                                                  subjectInfos=subjectInfo,
                                                  modelInfos=modelInfo,
                                                  experimentalInfos=experimentalInfo)
    
    analysisFilter = CGM2analysis.AnalysisFilter()
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()
    if exportAnalysisC3dFlag:
        c3dAnalysisName = modelledFilenames[0][:-4]+"-Cycles" if len(modelledFilenames) == 1 else  "MultiTrials"
        analysisFilter.exportAnalysisC3d(c3dAnalysisName, path=DATA_PATH)

    if exportSpreadSheetFlag:
        spreadSheetName = modelledFilenames[0][:-4] if len(modelledFilenames) == 1 else  "MultiTrials"
        analysisFilter.exportBasicDataFrame(spreadSheetName, path=DATA_PATH)
        analysisFilter.exportAdvancedDataFrame(spreadSheetName, path=DATA_PATH)

    #---- GAIT PLOTTING FILTER
    #--------------------------------------------------------------------------
    if plotFlag:    
        plotBuilder = CGM2plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=True)
       
        # Filter
        pf = CGM2plot.PlottingFilter()
        pf.setBuilder(plotBuilder)

        if normativeDataDict["Author"] == "Schwartz2008":
            chosenModality = normativeDataDict["Modality"]
            pf.setNormativeDataProcedure(CGM2normdata.Schwartz2008_normativeDataBases(chosenModality)) # modalites : "Very Slow" ,"Slow", "Free", "Fast", "Very Fast"
        elif normativeDataDict["Author"] == "Pinzone2014":
            chosenModality = normativeDataDict["Modality"]
            pf.setNormativeDataProcedure(CGM2normdata.Pinzone2014_normativeDataBases(chosenModality)) # modalites : "Center One" ,"Center Two"

        pf.setPath(DATA_PATH)
        pdfnameSuffix = modelledFilenames[0][:-4] if len(modelledFilenames) == 1 else  "MultiTrials"
        pf.setPdfSuffix(pdfnameSuffix)
        pf.plot(consistencyOnly=False)

        #os.startfile(DATA_PATH+"consistencyKinematics_"+ gaitFilenameLabelled[:-4]+".pdf")
        #os.startfile(DATA_PATH+"consistencyKinetics_"+ gaitFilenameLabelled[:-4]+".pdf")


        