# -*- coding: utf-8 -*-
import logging


import pyCGM2


import pdb
import pandas as pd
import numpy as np

class Pinzone2014_normativeDataBases(object):

    def __init__(self,centre):
        """ 
        **Description :** Constructor of Pinzone2014_normativeDataBases 
        

        :Parameters:
             - `centre` (str) - two choices : CentreOne or CentreTwo 
 
        **Usage**
       
        .. code:: python

            from pyCGM2.Report import normativeDatabaseProcedure
            nd = normativeDatabaseProcedure.Pinzone2014_normativeDataBases("CentreOne")
            nd.constructNormativeData() # this function 
            nd.data # dictionnary with all parameters extracted from the dataset CentreOne reference in Pinzone2014 
 
 
        """
        
        self.m_filename = pyCGM2.CONFIG.NORMATIVE_DATABASE_PATH+"Pinzone 2014\\Formatted- Pinzone2014.xlsx"
        self.m_centre = centre
        self.data = dict()
    
    def __setDict(self,dataframe,JointLabel,axisLabel, dataType):
        """ populate an item of the member dictionnary (data) 
         
        """

        if self.m_centre == "CentreOne":
            meanLabel = "CentreOneAverage"
            sdLabel = "CentreOneSD"

        elif self.m_centre == "CentreTwo":
            meanLabel = "CentreTwoAverage"
            sdLabel = "CentreTwoSD"
        else:
            raise Exception("[pyCGM2] - dont find Pinzone Normative data centre")

        if dataType == "Angles":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Angle == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T           
            data_X=dataframe[(dataframe.Angle == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T    

        if dataType == "Moments":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Moment == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T*1000.0           
            data_X=dataframe[(dataframe.Moment == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T*1000.0    

        if dataType == "Powers":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Power == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T           
            data_X=dataframe[(dataframe.Power == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T

    
    def constructNormativeData(self):
        
        """ 
            **Description :**  Read initial xls file and construct the member dictionnary (data)
        """
        
        angles =pd.read_excel(self.m_filename,sheetname = "Angles")        
        moments =pd.read_excel(self.m_filename,sheetname = "Moments")
        powers =pd.read_excel(self.m_filename,sheetname = "Powers")                        
        
        self.__setDict(angles,"Pelvis.Angles",["Pelvis Ant/Pst", "Pelvic Up/Dn", "Pelvic Int/Ext" ], "Angles")
        self.__setDict(angles,"Hip.Angles",["Hip Flx/Ext", "Hip Add/Abd", "Hip Int/Ext" ], "Angles")
        self.__setDict(angles,"Knee.Angles",["Knee Flx/Ext",None, None], "Angles")
        self.__setDict(angles,"Ankle.Angles",["Ankle Dor/Pla", None, "Foot Int/Ext"], "Angles")

        self.__setDict(moments,"Hip.Moment",["Hip Extensor Moment ", "Hip Abductor Moment ", None ], "Moments")
        self.__setDict(moments,"Knee.Moment",["Knee Extensor Moment ", "Knee Abductor Moment ", None ], "Moments")
        self.__setDict(moments,"Ankle.Moment",["Plantarflexor Moment ", None, "Ankle Rotation Moment"], "Moments")
        
        self.__setDict(powers,"Hip.Power",[ None, None,"Hip Power" ], "Powers")
        self.__setDict(powers,"Knee.Power",[None, None,"Knee Power"], "Powers")
        self.__setDict(powers,"Ankle.Power",[None, None,"Ankle Power"], "Powers")
            
            
class Schwartz2008_normativeDataBases(object):


    def __init__(self,speed):

        """ 
        **Description :** Constructor of Schwartz2008_normativeDataBases

        :Parameters:
               - `speed` (str) -  choices : VerySlow, Slow, Free, Fast, VeryFast 
 
        **usage**
       
        .. code:: python

            from pyCGM2.Report import normativeDatabaseProcedure
            nd = normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free")
            nd.constructNormativeData() # this function 
            nd.data # dictionnary with all parameters extracted from the dataset CentreOne reference in Pinzone2014 
 
 
        """

        self.m_filename = pyCGM2.CONFIG.NORMATIVE_DATABASE_PATH+"Schwartz 2008\\Formatted- Schwartz2008.xlsx"        
        
        self.m_speedModality = speed
        self.data = dict()

    def __setDict(self,dataframe,JointLabel,axisLabel, dataType):
        """ 
            Populate an item of the member dictionnary (data) 
         
        """        
        
        
        if self.m_speedModality == "VerySlow":
            meanLabel = "VerySlowMean"
            sdLabel = "VerySlowSd"
        elif self.m_speedModality == "Slow":
            meanLabel = "SlowMean"
            sdLabel = "SlowSd"
        elif self.m_speedModality == "Free":
            meanLabel = "FreeMean"
            sdLabel = "FreeSd"
        elif self.m_speedModality == "Fast":
            meanLabel = "FastMean"
            sdLabel = "FastSd"
        elif self.m_speedModality == "VeryFast":
            meanLabel = "VeryFastMean"
            sdLabel = "VeryFastSd"

                            
                
        if dataType == "Angles":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Angle == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
            
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T           
            data_X=dataframe[(dataframe.Angle == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T    

        if dataType == "Moments":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Moment == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
                   
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T*1000.0           
            data_X=dataframe[(dataframe.Moment == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T*1000.0    

        if dataType == "Powers":
            self.data[JointLabel]= dict() 
            data_X=dataframe[(dataframe.Power == axisLabel[0])][meanLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].as_matrix() if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][meanLabel].as_matrix() if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T           
            data_X=dataframe[(dataframe.Power == axisLabel[0])][sdLabel].as_matrix() if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].as_matrix()  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][sdLabel].as_matrix()  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T
        
        
    def constructNormativeData(self):
        """ 
            **Description :**  Read initial xls file and construct the member dictionnary (data)
        """

        angles =pd.read_excel(self.m_filename,sheetname = "Joint Rotations")        
        moments =pd.read_excel(self.m_filename,sheetname = "Joint Moments")
        powers =pd.read_excel(self.m_filename,sheetname = "Joint Power")                        
        
        
        self.__setDict(angles,"Pelvis.Angles",["Pelvic Ant/Posterior Tilt", "Pelvic Up/Down Obliquity", "Pelvic Int/External Rotation" ], "Angles")
        self.__setDict(angles,"Hip.Angles",["Hip Flex/Extension", "Hip Ad/Abduction", "Hip Int/External Rotation" ],"Angles")
        self.__setDict(angles,"Knee.Angles",["Knee Flex/Extension","Knee Ad/Abduction", "Knee Int/External Rotation"], "Angles")
        self.__setDict(angles,"Ankle.Angles",["Ankle Dorsi/Plantarflexion", None, "Foot Int/External Progression"], "Angles")

        self.__setDict(moments,"Hip.Moment",["Hip Ext/Flexion", "Hip Ab/Adduction", None ], "Moments")
        self.__setDict(moments,"Knee.Moment",["Knee Ext/Flexion", "Knee Ab/Adduction", None ], "Moments")
        self.__setDict(moments,"Ankle.Moment",["Ankle Dorsi/Plantarflexion", None, None], "Moments")
        
        self.__setDict(powers,"Hip.Power",[ None, None,"Hip" ], "Powers")
        self.__setDict(powers,"Knee.Power",[None, None,"Knee"], "Powers")
        self.__setDict(powers,"Ankle.Power",[None, None,"Ankle"], "Powers")