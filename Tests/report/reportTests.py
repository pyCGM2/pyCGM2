# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:54:18 2017

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

from pyCGM2.Processing import cycle,analysis
from pyCGM2.Report import plot,normativeDatabaseProcedure
from pyCGM2.Tools import trialTools

class PlotTest(): 

    @classmethod
    def OneDescriptiveGaitPlot(cls):
         # ----DATA-----        
        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\plot\\gaitPlot\\"
        reconstructedFilenameLabelledNoExt ="gait Trial 03 - viconName"  
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"        
    
        logging.info("data Path: "+ DATA_PATH)    
        logging.info( "reconstructed file: "+ reconstructedFilenameLabelled)
        
        
        modelledFilenames = [reconstructedFilenameLabelled]        
        
        #---- GAIT CYCLES FILTER PRELIMARIES
        #--------------------------------------------------------------------------
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
    
        # ----INFOS-----        
        modelInfo=None  
        subjectInfo=None
        experimentalInfo=None    
    
        pointLabelSuffix =""
        pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix
    
        kinematicLabelsDict ={ 'Left': ["LHipAngles"+pointLabelSuffixPlus,"LKneeAngles"+pointLabelSuffixPlus,"LAnkleAngles"+pointLabelSuffixPlus,"LFootProgressAngles"+pointLabelSuffixPlus,"LPelvisAngles"+pointLabelSuffixPlus],
                        'Right': ["RHipAngles"+pointLabelSuffixPlus,"RKneeAngles"+pointLabelSuffixPlus,"RAnkleAngles"+pointLabelSuffixPlus,"RFootProgressAngles"+pointLabelSuffixPlus,"RPelvisAngles"+pointLabelSuffixPlus] }
    
        kineticLabelsDict ={ 'Left': ["LHipMoment"+pointLabelSuffixPlus,"LKneeMoment"+pointLabelSuffixPlus,"LAnkleMoment"+pointLabelSuffixPlus, "LHipPower"+pointLabelSuffixPlus,"LKneePower"+pointLabelSuffixPlus,"LAnklePower"+pointLabelSuffixPlus],
                        'Right': ["RHipMoment"+pointLabelSuffixPlus,"RKneeMoment"+pointLabelSuffixPlus,"RAnkleMoment"+pointLabelSuffixPlus, "RHipPower"+pointLabelSuffixPlus,"RKneePower"+pointLabelSuffixPlus,"RAnklePower"+pointLabelSuffixPlus]}
    
    
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)
        
        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.build()

        # --- plot ---
        fig = plt.figure()
        ax = plt.axes() 

        plot.gaitDescriptivePlot(ax, analysisFilter.analysis.kinematicStats, 
                                 "LHipAngles","Left", 
                                 "RHipAngles","Right",0,
                                 "Pelvis Tilt", ylabel = " angle (deg)")

        fig = plt.figure()
        ax = plt.axes() 

        plot.gaitConsistencyPlot(ax, analysisFilter.analysis.kinematicStats, 
                                 "LHipAngles","Left", 
                                 "RHipAngles","Right",0,
                                 "Pelvis Tilt", ylabel = " angle (deg)")


    @classmethod
    def GaitPlotPanelWithLimits(cls):
        
        
        # ----DATA-----        
        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\plot\\gaitPlot\\"
        reconstructedFilenameLabelledNoExt ="gait Trial 03 - viconName"  
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"        
    
        logging.info("data Path: "+ DATA_PATH)    
        logging.info( "reconstructed file: "+ reconstructedFilenameLabelled)
        
        
        modelledFilenames = [reconstructedFilenameLabelled]        
        
        #---- GAIT CYCLES FILTER PRELIMARIES
        #--------------------------------------------------------------------------
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
    
        # ----INFOS-----        
        modelInfo=None  
        subjectInfo=None
        experimentalInfo=None    
    
        pointLabelSuffix =""
        pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix
    
        kinematicLabelsDict ={ 'Left': ["LHipAngles"+pointLabelSuffixPlus,"LKneeAngles"+pointLabelSuffixPlus,"LAnkleAngles"+pointLabelSuffixPlus,"LFootProgressAngles"+pointLabelSuffixPlus,"LPelvisAngles"+pointLabelSuffixPlus],
                        'Right': ["RHipAngles"+pointLabelSuffixPlus,"RKneeAngles"+pointLabelSuffixPlus,"RAnkleAngles"+pointLabelSuffixPlus,"RFootProgressAngles"+pointLabelSuffixPlus,"RPelvisAngles"+pointLabelSuffixPlus] }
    
        kineticLabelsDict ={ 'Left': ["LHipMoment"+pointLabelSuffixPlus,"LKneeMoment"+pointLabelSuffixPlus,"LAnkleMoment"+pointLabelSuffixPlus, "LHipPower"+pointLabelSuffixPlus,"LKneePower"+pointLabelSuffixPlus,"LAnklePower"+pointLabelSuffixPlus],
                        'Right': ["RHipMoment"+pointLabelSuffixPlus,"RKneeMoment"+pointLabelSuffixPlus,"RAnkleMoment"+pointLabelSuffixPlus, "RHipPower"+pointLabelSuffixPlus,"RKneePower"+pointLabelSuffixPlus,"RAnklePower"+pointLabelSuffixPlus]}
    
    
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=modelInfo,
                                                      experimentalInfos=experimentalInfo)
        
        analysisFilter = analysis.AnalysisFilter()
        analysisFilter.setBuilder(analysisBuilder)
        analysisFilter.build()
    
       
        plotBuilder = plot.GaitAnalysisPlotBuilder(analysisFilter.analysis , kineticFlag=flag_kinetics, pointLabelSuffix= pointLabelSuffix)
        plotBuilder.setNormativeDataProcedure(normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free"))  
        plotBuilder.setConsistencyOnly(True)
        plotBuilder.setLimits("Left.Knee.Angles","Flex" , [10,20,30])
        plotBuilder.setLimits("Right.Ankle.Moment","Pla" , [1000])              
       
       
        # Filter
        pf = plot.PlottingFilter()
        pf.setBuilder(plotBuilder)
        pf.setPath(DATA_PATH)
 
        pf.setPdfName("TEST")
        pf.plot()
        

        

if __name__ == "__main__":

    plt.close("all")  
  
    PlotTest.OneDescriptiveGaitPlot()    
    PlotTest.GaitPlotPanelWithLimits()    

    