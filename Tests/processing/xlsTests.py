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

from pyCGM2 import smartFunctions

class xlsExportTest(): 

    @classmethod
    def gaitXlsExport_oneFile(cls):
        
        # ----DATA-----        

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\analysis\\gait\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d" ] 
                
        # ----INFOS-----        
        modelInfo=None  
        subjectInfo=None
        experimentalInfo=None 

        normativeDataSet=dict()
        normativeDataSet["Author"] = "Schwartz2008"
        normativeDataSet["Modality"] = "Free"           
        
        smartFunctions.gaitProcessing_cgm1 (modelledFilenames, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo,
                         pointLabelSuffix = "",
                         plotFlag= False, 
                         exportBasicSpreadSheetFlag = True,
                         exportAdvancedSpreadSheetFlag = True,
                         exportAnalysisC3dFlag = False,
                         consistencyOnly = False,
                         normativeDataDict = normativeDataSet,
                         name_out=None,
                         DATA_PATH_OUT= None,
                         longitudinal_axis=None,
                         lateral_axis = None)
                         

    @classmethod
    def gaitXlsExport_twoFiles(cls):
        
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
        
        smartFunctions.gaitProcessing_cgm1 (modelledFilenames, DATA_PATH, 
                         modelInfo, subjectInfo, experimentalInfo,
                         pointLabelSuffix = "",
                         plotFlag= False, 
                         exportBasicSpreadSheetFlag = True,
                         exportAdvancedSpreadSheetFlag = True,
                         exportAnalysisC3dFlag = False,
                         consistencyOnly = False,
                         normativeDataDict = normativeDataSet,
                         name_out=None,
                         DATA_PATH_OUT= None,
                         longitudinal_axis=None,
                         lateral_axis = None)

if __name__ == "__main__":

    plt.close("all")  
  
    #xlsExportTest.gaitXlsExport_oneFile()
    xlsExportTest.gaitXlsExport_twoFiles()   