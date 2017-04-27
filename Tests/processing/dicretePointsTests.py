
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:55:11 2017

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import pdb
import logging


import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body

from pyCGM2.Report import plot,normativeDatabaseProcedure
from pyCGM2.Processing import cycle,analysis, discretePoints,exporter
from pyCGM2.Tools import trialTools

class BenedettiTest(): 

    @classmethod
    def kinematics(cls):
                
        # ----DATA-----        

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d","gait Trial 03 - viconName.c3d"  ] 
                
        # ----INFOS-----        
        modelInfo=None  
        subjectInfo=None
        experimentalInfo=None 

        normativeDataSet=dict()
        normativeDataSet["Author"] = "Schwartz2008"
        normativeDataSet["Modality"] = "Free"           
        

        pointLabelSuffix=""        
        
        # distinguishing trials for kinematic and kinetic processing                             
        # - kinematic Trials      
        kinematicTrials=[]
        kinematicFilenames =[]
        for kinematicFilename in modelledFilenames:
            kinematicFileNode = ma.io.read(str(DATA_PATH + kinematicFilename))
            kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            trialTools.sortedEvents(kinematicTrial)
    

            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgressionFromPoints(kinematicTrial,"LPSI","LASI","RPSI")
    
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


#        #---- GAIT ANALYSIS FILTER
#        #--------------------------------------------------------------------------
#
#        plotBuilder = plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=flag_kinetics, pointLabelSuffix= pointLabelSuffix)
#        plotBuilder.setNormativeDataProcedure(normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free"))  
#        plotBuilder.setConsistencyOnly(True)
#              
#        # Filter
#        pf = plot.PlottingFilter()
#        pf.setBuilder(plotBuilder)
#        pf.setPath(DATA_PATH)
# 
#        pf.setPdfName("TEST")
#        pf.plot()


        #---- DISCRETE POINT FILTER
        #--------------------------------------------------------------------------

        # Benedetti Processing
        dpProcedure = discretePoints.BenedettiProcedure()
        dpf = discretePoints.DiscretePointsFilter(dpProcedure, analysisFilter.analysis)
        benedettiDataFrame = dpf.getOutput()

        xlsExport = exporter.XlsExportFilter()
        xlsExport.setDataFrames([benedettiDataFrame])
        xlsExport.exportDataFrames("discretePoints", path=DATA_PATH)
        
        
class MaxMinTest(): 

    @classmethod
    def kinematics(cls):
                
        # ----DATA-----        

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 01 - viconName.c3d","gait Trial 03 - viconName.c3d"  ] 
                
        # ----INFOS-----        
        modelInfo=None  
        subjectInfo=None
        experimentalInfo=None 

        normativeDataSet=dict()
        normativeDataSet["Author"] = "Schwartz2008"
        normativeDataSet["Modality"] = "Free"           
        

        pointLabelSuffix=""        
        
        # distinguishing trials for kinematic and kinetic processing                             
        # - kinematic Trials      
        kinematicTrials=[]
        kinematicFilenames =[]
        for kinematicFilename in modelledFilenames:
            kinematicFileNode = ma.io.read(str(DATA_PATH + kinematicFilename))
            kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            trialTools.sortedEvents(kinematicTrial)
    

            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgressionFromPoints(kinematicTrial,"LPSI","LASI","RPSI")
    
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


#        #---- GAIT ANALYSIS FILTER
#        #--------------------------------------------------------------------------
#
#        plotBuilder = plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=flag_kinetics, pointLabelSuffix= pointLabelSuffix)
#        plotBuilder.setNormativeDataProcedure(normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free"))  
#        plotBuilder.setConsistencyOnly(True)
#              
#        # Filter
#        pf = plot.PlottingFilter()
#        pf.setBuilder(plotBuilder)
#        pf.setPath(DATA_PATH)
# 
#        pf.setPdfName("TEST")
#        pf.plot()


        #---- DISCRETE POINT FILTER
        #--------------------------------------------------------------------------

        # Benedetti Processing
        dpProcedure = discretePoints.MaxMinProcedure()
        dpf = discretePoints.DiscretePointsFilter(dpProcedure, analysisFilter.analysis)
        dataFrame = dpf.getOutput()

        xlsExport = exporter.XlsExportFilter()
        xlsExport.setDataFrames(dataFrame)
        xlsExport.exportDataFrames("discretePointsMaxMin", path=DATA_PATH)
        
if __name__ == "__main__":

    plt.close("all")  
  
    #BenedettiTest.kinematics()
    MaxMinTest.kinematics()