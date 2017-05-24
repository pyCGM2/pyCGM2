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

from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Processing import cycle,analysis,scores,exporter,c3dManager
from pyCGM2.Report import normativeDatabaseProcedure
from pyCGM2.Tools import trialTools




class GpsTest(): 

    @classmethod
    def GpsCGM1Test(cls):
         # ----DATA-----        
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH+"operations\\analysis\\gps\\"

        reconstructedFilenameLabelledNoExt ="gait Trial 03 - viconName"  
        reconstructedFilenameLabelled = reconstructedFilenameLabelledNoExt+".c3d"        
    
        logging.info("data Path: "+ DATA_PATH)    
        logging.info( "reconstructed file: "+ reconstructedFilenameLabelled)
        
        
        modelledFilenames = [reconstructedFilenameLabelled]        
        
        #---- c3d manager
        #--------------------------------------------------------------------------
                
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)
        cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
        cmf.enableEmg(False)
        trialManager = cmf.generate()
        
    
    
        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=trialManager.spatioTemporal["Trials"],
                                                   kinematicTrials = trialManager.kinematic["Trials"],
                                                   kineticTrials = trialManager.kinetic["Trials"],
                                                   emgTrials=trialManager.emg["Trials"])    
                
        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()
    
    
        #---- GAIT ANALYSIS FILTER
        #--------------------------------------------------------------------------
    
        # ----INFOS-----        
        modelInfo={"type":"S01"}  
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

        ## --- GPS ----
        gps =scores.CGM1_GPS()
        ndp = normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free")
        
        scf = scores.ScoreFilter(gps,analysisFilter.analysis, ndp)
        scf.compute()

        xlsExport = exporter.XlsExportFilter()
        xlsExport.setAnalysisInstance(analysisFilter.analysis)
        xlsExport.setConcreteAnalysisBuilder(analysisBuilder)
        xlsExport.exportAdvancedDataFrame("gpsTest", path=DATA_PATH)




        

if __name__ == "__main__":

    plt.close("all")  
  
    GpsTest.GpsCGM1Test()    


    