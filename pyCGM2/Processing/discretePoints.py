# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import pdb
import pandas as pd
from pyCGM2.Tools import exportTools
from collections import OrderedDict
from pyCGM2.Signal.detect_peaks import detect_peaks    


#            x= normalizedCycleValues["values"][0][:,2]
#            plt.plot(x)            
#            plt.plot(frameMin,x[frameMin], 'r+')
#            print x[frameMin]


import matplotlib.pyplot as plt
# --- FILTER ----


class DiscretePointsFilter(object):
    """
    """

    def __init__(self, discretePointProcedure, analysis):

        self.m_procedure = discretePointProcedure      
        self.m_analysis=analysis
        
        self.dataframe =  None
         

    def getOutput(self):
        self.dataframe = self.m_procedure.detect(self.m_analysis)
        
        return self.dataframe

    
        
        

# --- PROCEDURE ----
class BenedettiProcedure(object):

    NAME = "Benedetti"

       
    def __init__(self,pointSuffix=None):

        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""
        
    def detect (self,analysisInstance):

        # self.__detectTest(analysisInstance,"RHipMoment","Right") # TEST
        
        dataframes = list()
        # Left
        dataframes.append( self.__getPelvis_kinematics(analysisInstance,"LPelvisAngles","Left"))

        dataframes.append( self.__getHip_kinematics(analysisInstance,"LPelvisAngles","Left"))
        dataframes.append( self.__getHip_kinematics(analysisInstance,"LHipAngles","Left"))
        dataframes.append( self.__getKnee_kinematics(analysisInstance,"LKneeAngles","Left"))
        dataframes.append( self.__getAnkle_kinematics(analysisInstance,"LAnkleAngles","Left")) 

        dataframes.append( self.__getHip_kinetics(analysisInstance,"LHipMoment","Left"))
        dataframes.append( self.__getKnee_kinetics(analysisInstance,"LKneeMoment","Left"))
        dataframes.append( self.__getAnkle_kinetics(analysisInstance,"LAnkleMoment","Left"))

        # Right
        dataframes.append( self.__getPelvis_kinematics(analysisInstance,"RPelvisAngles","Right"))

        dataframes.append( self.__getHip_kinematics(analysisInstance,"RHipAngles","Right"))
        dataframes.append( self.__getKnee_kinematics(analysisInstance,"RKneeAngles","Right"))
        dataframes.append( self.__getAnkle_kinematics(analysisInstance,"RAnkleAngles","Right")) 

        dataframes.append( self.__getHip_kinetics(analysisInstance,"RHipMoment","Right"))
        dataframes.append( self.__getKnee_kinetics(analysisInstance,"RKneeMoment","Right"))
        dataframes.append( self.__getAnkle_kinetics(analysisInstance,"RAnkleMoment","Right"))

        return pd.concat(dataframes)

    def __construcPandasSerie(self,pointLabel,context, axis, cycleIndex, 
                              discretePointProcedure,discretePointLabel,discretePointValue,discretePointDescription,
                              comment):
        iDict = OrderedDict([('Label', pointLabel), 
                     ('Context', context), 
                     ('Axis', axis),
                     ('Cycle', cycleIndex),
                     ('DiscretePointProcedure', discretePointProcedure),
                     ('DiscretePointLabel', discretePointLabel),
                     ('DiscretePointValue', discretePointValue),
                     ('DiscretePointDescription', discretePointDescription),
                     ('Comment', comment)])
        return pd.Series(iDict)

    # TEST -----------------
    def __detectTest(self,analysisInstance,pointLabel,context):

#        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
#        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
#        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']



        # experiment with detect_peaks, marcos's function
        i=1     

        fullValues = normalizedCycleValues["values"][i][:,1]
        
        begin = 0
        end  = 101
        values = normalizedCycleValues["values"][i][begin:end,1]
        indexes = detect_peaks(values,show = False, valley=False)
        plt.figure()
        plt.plot(fullValues)
        plt.axvline(int(stanceValues[i]),color='r',ls='dashed')
        for it in indexes:
            plt.plot(begin+it,fullValues[begin+it],"+r",mew=2, ms=8)


  
    # /TEST -----------------
        
    def __getPelvis_kinematics(self,analysisInstance,pointLabel,context):
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()        
        
        #---min rotation sagital plane  
        label = ["HR1","THR1"]
        axis = "X"
        desc = ["min rotation sagital plane","frame of min rotation sagital plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][:,0])
            frame = np.argmin(normalizedCycleValues["values"][i][:,0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---min rot coronal plane ( erreur dans la table 1 de benedetti)
        label = ["HR2","THR2"]
        axis = "Y"
        desc = ["min rot coronal plane","frame of min rot coronal plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][:,1])
            frame = np.argmin(normalizedCycleValues["values"][i][:,1])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot coronal plane
        label = ["HR3","THR3"]
        axis = "Y"
        desc = ["max rot coronal plane","frame of max rot coronal plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1])
            frame = np.argmax(normalizedCycleValues["values"][i][:,1])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #--- max rot transverse plane
        label = ["HR4","THR4"]
        axis = "Y"
        desc = ["max rot transverse plane","frame of max rot transverse plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1])
            frame = np.argmax(normalizedCycleValues["values"][i][:,1])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        return pd.DataFrame(series)        

    def __getHip_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()        
        
        #---flexion at heel strike                 
        label = "H1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response(H2)                
        label = "H2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---extension max in stance(H3-TH3)  
        label = ["H3","TH3"]
        axis = "X"
        desc = ["extension max in stance","frame of extension max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

              
       


        #---flexion at toe-off (H4)             
        label = "H4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


                                  
            
        #---max flexion in swing(H5-TH5)                
        label = ["H5","TH5"]
        axis = "X"
        desc = ["max flexion in swing","frame of max flexion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         


        #---total sagital plane excursion(H6))               
        label = "H6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---total coronal plane excursion(H7))               
        label = "H7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1]) - np.min(normalizedCycleValues["values"][i][:,1]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---max adduction in stance(H8-TH8))                           
        label = ["H8","TH8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



       

        #---max abd in swing(H9-TH9))
        label = ["H9","TH9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

               

        #---total transverse plane excursion(H10))               
        label = "H10"
        axis = "Z"
        desc = "total transverse plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,2]) - np.min(normalizedCycleValues["values"][i][:,2]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---max rot int in stance(H10-TH10))               
        label = ["H11","TH11"]
        axis = "Z"
        desc = ["max rot int in stance","frame of max rot int in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2]) 
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot ext in swing(H11-TH11))               
        label = ["H12","TH12"]
        axis = "Z"
        desc = ["max rot ext in swing","frame of max rot ext in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2]) 
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


     
        return pd.DataFrame(series)                                  


    def __getKnee_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        #---flexion at heel strike                 
        label = "K1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response                
        label = "K2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---extension max in stance 
        label = ["K3","TK3"]
        axis = "X"
        desc = ["extension max in stance","frame of extension max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---flexion at toe-off             
        label = "K4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

            
        #---max flexion in swing               
        label = ["K5","TK5"]
        axis = "X"
        desc = ["max flexion in swing","frame of max flexion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         


        #---total sagital plane excursion               
        label = "K6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---total coronal plane excursion               
        label = "K7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1]) - np.min(normalizedCycleValues["values"][i][:,1]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---max adduction in stance(H8-TH8))                           
        label = ["K8","TK8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



       

        #---max abd in swing(H9-TH9))
        label = ["K9","TK9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

               

        #---total transverse plane excursion               
        label = "K10"
        axis = "Z"
        desc = "total transverse plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,2]) - np.min(normalizedCycleValues["values"][i][:,2]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #--max rot int in stance               
        label = ["K11","TK11"]
        axis = "Z"
        desc = ["max rot int in stance","frame of max rot int in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2]) 
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot ext in swing               
        label = ["K12","TK12"]
        axis = "Z"
        desc = ["max rot ext in swing","frame of max rot ext in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2]) 
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


     
        return pd.DataFrame(series)
        
    def __getAnkle_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        #---flexion at heel strike                 
        label = "A1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response                
        label = "A2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---dorsi flex max in stance
        label = ["A3","TA3"]
        axis = "X"
        desc = ["dorsi flex max in stance","frame of dorsi flex max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #---flexion at toe-off              
        label = "A4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #--- 5- max plant flexion in swing ( error in benedetti table) !!! 

        #  rule : 
        #    - find the first peak inferior to value at TO from pre-swing(ie TO reduced by 10 frames) 
        # note : 
        #    - too dependant to detection of the TO frame. Maximal plantar flexion can occurs before Toe off !
        #   - use detect_peak 

        label = ["A5","TA5"]
        axis = "X"
        desc = ["max plant flexion in swing","frame of max plant flexion in swing"]
        
        for i in range(0,len(normalizedCycleValues["values"])):

            threshold = 10
            frameTO = int(stanceValues[i])
            beginFrame = frameTO-threshold
            toeOffValue = normalizedCycleValues["values"][i][frameTO,0]

            values = normalizedCycleValues["values"][i][:,0]
            valuesFromPreswing = normalizedCycleValues["values"][i][beginFrame:101,0]
            indexes = detect_peaks(valuesFromPreswing, valley=True)

            # rule application
            frame = "NA"
            value = "NA"
            comment =""
            if indexes != []:
                for ind in indexes:
                    if values[ind]<toeOffValue: 
                        frame = beginFrame + int(ind)
                        value = values[frame]
                        if frame < frameTO: comment = " warning : value before TO" 
                        break
                    else:
                        frame = "NA"
                        value = "NA"
                        comment = "no peak found"
            

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               comment)
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               comment)
            series.append(serie)

        #---6 total sagital plane excursion !!
        # search for max/min without regard to phases
        # note : min might be different from A5 if min occurs before preswing (TO-10 frames) 

        label = "A6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #--- 7 total coronal plane excursion !!!
        # note : 
        #   -max in stance (ie. A7) minus min in swing (ie. A8)                              
        label = "A7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) - np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)
 
        #--- 8 max inversion in stance                           
        label = ["A8","TA8"]
        axis = "Y"
        desc = ["max inversion in stance","frame of max inversion in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #-- 9 max eversion in swing
        label = ["A9","TK9"]
        axis = "Y"
        desc = ["max eversion in swing","frame of max eversion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

     
        return pd.DataFrame(series)  


    def __getHip_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        

        #---1-max and 2-min extensor moments
        
        for i in range(0,len(normalizedCycleValues["values"])):
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            valueMax = np.max(normalizedCycleValues["values"][i][0:frameMin,0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:frameMin,0])

            label = ["HM1","THM1"]
            axis = "X"
            desc = ["max flex moment", "frame of max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[1],
                                               "")
                                               
            series.append(serie)

            label = ["HM2","THM2"]
            axis = "X"
            desc = ["max ext moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[1],
                                               "")
                                               
            series.append(serie)
        
        
        #---3 first and 4 second abductor moments !!
        # rule : 
        #  - find  first max during the first mid stance
        # - find the second max from the first max frame
        # note : 
        #  - apparently Benedetti aductor curve is not conventional, compared wth plot from gait books

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax1 = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])
            frameMax1 = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])

            valueMax2 = np.max(normalizedCycleValues["values"][i][frameMax1:int(stanceValues[i]),1])
            frameMax2 = frameMax1 + np.argmax(normalizedCycleValues["values"][i][frameMax1:int(stanceValues[i]),1])

            label = ["HM3","THM3"]
            axis = "Y"
            desc = ["first max abductor moment", "frame of the first max abductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax1,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax1,desc[1],
                                               "")
                                               
            series.append(serie)

            label = ["HM4","THM4"]
            axis = "Y"
            desc = ["second max abductor moment", "frame of the second max abductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax2,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax2,desc[1],
                                               "")
                                               
            series.append(serie)

        
        #---5 -min and 6-max rotation moment
        for i in range(0,len(normalizedCycleValues["values"])):
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            label = ["HM5","THM5"]
            axis = "Z"
            desc = ["max ext rot moment", "frame of max ext rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[1],
                                               "")
                                               
            series.append(serie)

            label = ["HM6","THM6"]
            axis = "Z"
            desc = ["max int rot moment", "frame of max int rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[1],
                                               "")
                                               
            series.append(serie)

        return pd.DataFrame(series)
    
    def __getKnee_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        
        #---1 fist flexor extensor, 2- max extensor, 3-second max flexor moment !!!
        # rule : 
        #  - detect extensor max firstly and find first and second max flexor subsequently
        # note :
        #  - benededdi confuse flex and extensor moment apparently
        
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            if frameMax !=0:
                valueFistMin = np.min(normalizedCycleValues["values"][i][0:int(frameMax),0])
                frameFistMin = np.argmin(normalizedCycleValues["values"][i][0:int(frameMax),0])
                commentFistMin =""
            else:
                valueFistMin ="NA"
                frameFistMin = "NA"
                commentFistMin = "Warning first extensor dtected at frame 0" 
                
                

            valueSecondMin = np.min(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])
            frameSecondMin = int(frameMax) + np.argmin(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])
 

            label = ["KM1","TKM1"]
            axis = "X"
            desc = ["first max flex moment", "frame of first max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueFistMin,desc[0],
                                               commentFistMin)
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameFistMin,desc[0],
                                               commentFistMin)
                                               
            series.append(serie)

            label = ["KM2","TKM2"]
            axis = "X"
            desc = ["max extensor moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMax,desc[0],
                                               "")
                                               
            series.append(serie)


            label = ["KM3","TKM3"]
            axis = "X"
            desc = ["second max flex moment", "frame of second max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueSecondMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameSecondMin,desc[0],
                                               "")
                                               
            series.append(serie)

        #---4 max adductor, 5- first max abd , 6- second max abd  moment
        # rule :
        #   - find first max abd on mid stance
        #   - find max add from 0 to first max abd
        #   - find second abd max from max abd to TO

        # note :
        #  - Benedetti illustred an opposite trace

        for i in range(0,len(normalizedCycleValues["values"])):

            valueFirstMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])
            frameFirstMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])

            
            if frameFirstMax !=0:
                valueMin = np.min(normalizedCycleValues["values"][i][0:int(frameFirstMax),1])
                frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(frameFirstMax),1])
                commentMin =""
            else:
                valueMin,frameMin ="NA","NA"
                commentMin = "warning first max detected at 0"
                
            
            valueSecondMax = np.max(normalizedCycleValues["values"][i][int(frameFirstMax):int(stanceValues[i]),1])
            frameSecondMax = int(frameFirstMax) + np.argmax(normalizedCycleValues["values"][i][int(frameFirstMax):int(stanceValues[i]),1])
 

            label = ["KM4","TKM4"]
            axis = "Y"
            desc = ["max add moment", "frame of max add moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               commentMin)
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMin,desc[0],
                                               commentMin)
                                               
            series.append(serie)
            
            
            label = ["KM5","TKM5"]
            axis = "Y"
            desc = ["first max abd moment", "frame of first max abd moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueFirstMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameFirstMax,desc[0],
                                               "")
                                               
            series.append(serie)

            label = ["KM6","TKM6"]
            axis = "Y"
            desc = ["second max abd moment", "frame of second max abd moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueSecondMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameSecondMax,desc[0],
                                               "")
                                               
            series.append(serie)

        #-- 7 max int, 8- max ext rotation moment
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])


            label = ["KM7","TKM7"]
            axis = "Z"
            desc = ["max int rot moment", "frame of max int rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMin,desc[0],
                                               "")
                                               
            series.append(serie)

            label = ["KM8","TKM8"]
            axis = "Z"
            desc = ["max ext rot moment", "frame of max ext rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMax,desc[0],
                                               "")
                                               
            series.append(serie)

        return pd.DataFrame(series)

            
    def __getAnkle_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        
        #---1 max flex , 2 max ext moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])


            label = ["AM1","TAM1"]
            axis = "X"
            desc = [" max flex moment", "frame of max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMin,desc[0],
                                               "")

            label = ["AM2","TAM2"]
            axis = "X"
            desc = [" max ext moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMax,desc[0],
                                               "")
                                               
        #--4 max eversor, 5 max inv moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])


            label = ["AM3","TAM3"]
            axis = "Y"
            desc = [" max ever moment", "frame of max ever moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMax,desc[0],
                                               "")

            label = ["AM4","TAM4"]
            axis = "Y"
            desc = [" max inv moment", "frame of max inv moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameMin,desc[0],
                                               "")
            
            series.append(serie)                                   
                                               
        return pd.DataFrame(series)


class MaxMinProcedure(object):

    NAME = "MaxMin"

       
    def __init__(self,pointSuffix=None):

        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""
        
    def detect (self,analysisInstance):

               
        dataframes = list()
        # Left
        dataframes.append( self.__getExtrema(analysisInstance,"LPelvisAngles","Left"))

        dataframes.append( self.__getExtrema(analysisInstance,"LPelvisAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LHipAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LKneeAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LAnkleAngles","Left")) 

        dataframes.append( self.__getExtrema(analysisInstance,"LHipMoment","Left", dataType = "Kinetics"))
        dataframes.append( self.__getExtrema(analysisInstance,"LKneeMoment","Left", dataType = "Kinetics"))
        dataframes.append( self.__getExtrema(analysisInstance,"LAnkleMoment","Left",dataType = "Kinetics"))

        # Right
        dataframes.append( self.__getExtrema(analysisInstance,"RPelvisAngles","Right"))

        dataframes.append( self.__getExtrema(analysisInstance,"RHipAngles","Right"))
        dataframes.append( self.__getExtrema(analysisInstance,"RKneeAngles","Right"))
        dataframes.append( self.__getExtrema(analysisInstance,"RAnkleAngles","Right")) 

        dataframes.append( self.__getExtrema(analysisInstance,"RHipMoment","Right",dataType = "Kinetics"))
        dataframes.append( self.__getExtrema(analysisInstance,"RKneeMoment","Right",dataType = "Kinetics"))
        dataframes.append( self.__getExtrema(analysisInstance,"RAnkleMoment","Right",dataType = "Kinetics"))

        return pd.concat(dataframes)

        
        return pd.concat(dataframes)

    def __construcPandasSerie(self,pointLabel,context, axis, cycleIndex, 
                              discretePointProcedure,discretePointLabel,discretePointValue,discretePointDescription,
                              comment):
        iDict = OrderedDict([('Label', pointLabel), 
                     ('Context', context), 
                     ('Axis', axis),
                     ('Cycle', cycleIndex),
                     ('DiscretePointProcedure', discretePointProcedure),
                     ('DiscretePointLabel', discretePointLabel),
                     ('DiscretePointValue', discretePointValue),
                     ('DiscretePointDescription', discretePointDescription),
                     ('Comment', comment)])
        return pd.Series(iDict)

        
    def __getExtrema(self,analysisInstance,pointLabel,context,dataType="Kinematics"):
        
        if dataType == "Kinematics":
            normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        if dataType == "Kinetics":
            normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]

        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()        
        
        #---min stance  
        label = ["minST","TminST"]
        axes = ["X","Y","Z"]
        
        for axInd in range(0,len(axes)):
                
            desc = ["min stance","frame of min stance"]
            for i in range(0,len(normalizedCycleValues["values"])):
                value = np.min(normalizedCycleValues["values"][i][:,axInd])
                frame = np.argmin(normalizedCycleValues["values"][i][:,axInd])
                
                serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                   int(i),
                                                   BenedettiProcedure.NAME,
                                                   label[0],value,desc[0],
                                                   "")
                series.append(serie)
                
                serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                   int(i),
                                                   BenedettiProcedure.NAME,
                                                   label[1],frame,desc[1],
                                                   "")
                series.append(serie) 

                #---max stance  
                label = ["maxST","TmaxST"]
                desc = ["max stance","frame of max stance"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.max(normalizedCycleValues["values"][i][:,axInd])
                    frame = np.argmax(normalizedCycleValues["values"][i][:,axInd])
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie)
        
                 #---min swing  
                label = ["minSW","TminSW"]
                desc = ["min swing","frame of min swing"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie) 
        
                #---max swing  
                label = ["maxSW","TmaxSW"]
                desc = ["max swing","frame of max swing"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)
                    
                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie)
                    
        return pd.DataFrame(series)
                                                                                            