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
         
         
    def find(self):
         self.dataframe = self.m_procedure.find(self.m_analysis)

        

# --- PROCEDURE ----
class BenedettiProcedure(object):

    NAME = "Benedetti"

    

       
    def __init__(self,pointSuffix=None):

        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""
        
    def find (self,analysisInstance):

        listOfseries = list()
        # Left
        series1 = self.__getHip_kinematics(analysisInstance,"LPelvisAngles","Left")
        series2 =  self.__getHip_kinematics(analysisInstance,"LHipAngles","Left")

#        listOfseries.append( self.__getHip_kinematics(analysisInstance,"LPelvisAngles","Left"))
#        listOfseries.append( self.__getHip_kinematics(analysisInstance,"LHipAngles","Left"))
#        listOfseries.append( self.__getKnee_kinematics(analysisInstance,"LKneeAngles","Left"))
#        listOfseries.append( self.__getAnkle_kinematics(analysisInstance,"LAnkleAngles","Left")) 
#
#        listOfseries.append( self.__getHip_kinetics(analysisInstance,"LHipMoment","Left"))
#        listOfseries.append( self.__getKnee_kinetics(analysisInstance,"LKneeMoment","Left"))
#        listOfseries.append( self.__getKnee_kinetics(analysisInstance,"LAnkleMoment","Left"))
#
#        # Right
#        listOfseries.append( self.__getHip_kinematics(analysisInstance,"RHipAngles","Right"))
#        listOfseries.append( self.__getKnee_kinematics(analysisInstance,"RKneeAngles","Right"))
#        listOfseries.append( self.__getAnkle_kinematics(analysisInstance,"RAnkleAngles","Right")) 
#
#        listOfseries.append( self.__getHip_kinetics(analysisInstance,"RHipMoment","Right"))
#        listOfseries.append( self.__getKnee_kinetics(analysisInstance,"RKneeMoment","Right"))
#        listOfseries.append( self.__getKnee_kinetics(analysisInstance,"RAnkleMoment","Right"))

        pdb.set_trace()
        return pd.DataFrame([listOfseries])

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

    def __getPelvis_kinematics(self,analysisInstance,pointLabel,context):
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()        
        
        # min rotation sagital plane (HR1,THR1) 
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        # min rot coronal plane ( erreur dans la table 1 de benedetti)
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        # max rot coronal plane
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        # max rot transverse plane
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        return series        

    def __getHip_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()        
        
        # flexion at heel strike (H1)                
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


        # flexion at loading response(H2)                
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

        # extension max in stance(H3-TH3)  
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

              
       


        # flexion at toe-off (H4)             
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


                                  
            
        # max flexion in swing(H5-TH5)                
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         


        # total sagital plane excursion(H6))               
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

        # total coronal plane excursion(H7))               
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

        #  max adduction in stance(H8-TH8))                           
        label = ["H8","TH8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



       

        #  max abd in swing(H9-TH9))
        label = ["H9","TH9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

               

        # total transverse plane excursion(H10))               
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


        #  max rot int in stance(H10-TH10))               
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         #  max rot ext in swing(H11-TH11))               
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


     
        return series                                  


    def __getKnee_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        # flexion at heel strike (H1)                
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


        # flexion at loading response(H2)                
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

        # extension max in stance(H3-TH3)  
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

              
       


        # flexion at toe-off (H4)             
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


                                  
            
        # max flexion in swing(H5-TH5)                
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         


        # total sagital plane excursion(H6))               
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

        # total coronal plane excursion(H7))               
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

        #  max adduction in stance(H8-TH8))                           
        label = ["K8","TK8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



       

        #  max abd in swing(H9-TH9))
        label = ["K9","TK9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1]) 
            
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

               

        # total transverse plane excursion(H10))               
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


        #  max rot int in stance(H10-TH10))               
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

         #  max rot ext in swing(H11-TH11))               
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


     
        return series
        
    def __getAnkle_kinematics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        # flexion at heel strike                 
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


        # flexion at loading response                
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

        # dorsi flex max in stance
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        # flexion at toe-off              
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


        # max plant flexion in swing ( error in benedetti table)

        label = ["K5","TK5"]
        axis = "X"
        desc = ["max plant flexion in swing","frame of max plant flexion in swing"]

        # experiment with detect_peaks, marcos's function
#        i=1        
#        values = normalizedCycleValues["values"][i][int(stanceValues[i]):101,0]
#        indexes = detect_peaks(-values, mpd=5, show=True)    
        
      
        for i in range(0,len(normalizedCycleValues["values"])):

            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        # total sagital plane excursion
        # note : max in stance minus min in swing               
        label = "A6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0]) - np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0]) 
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        # total coronal plane excursion
        # note : max in stance minus min in swing                              
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

        #  max inversion in stance                           
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #  max eversion in swing
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
            
            serie = self.__construcPandasSerie(label[1],context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

     
        return series  


    def __getHip_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        

        # max and min extensor moments
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
        
        
        # max and min abductor moments
        # note : 
        #  - first max detected during the first half of the stance phase  
        #  - second max detected during from the frame of the first min  

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMin1 = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])
            frameMin1 = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])

            valueMin2 = np.min(normalizedCycleValues["values"][i][frameMin1:int(stanceValues[i]),1])
            frameMin2 = frameMin1 + np.argmin(normalizedCycleValues["values"][i][frameMin1:int(stanceValues[i]),1])

            label = ["HM3","THM3"]
            axis = "Y"
            desc = ["first max adductor moment", "frame of the first max adductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin1,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin1,desc[1],
                                               "")
                                               
            series.append(serie)

            label = ["HM4","THM4"]
            axis = "Y"
            desc = ["second max adductor moment", "frame of the second max adductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin2,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin2,desc[1],
                                               "")
                                               
            series.append(serie)

        
        # max and min rotation moments
        for i in range(0,len(normalizedCycleValues["values"])):
            
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            valueMax = np.max(normalizedCycleValues["values"][i][frameMin:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][frameMin:int(stanceValues[i]),2])

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

    
    def __getKnee_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        
#        # experiment with detect_peaks, marcos's function
#        i=0        
#        values = normalizedCycleValues["values"][i][0:int(stanceValues[i]),0]
#        #plt.plot(values)
#        indexes = detect_peaks(values, show=True) 

        # extensor moment
        # note : - detect extensor max firstly and find first and second max flexor subsequently 
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            
            valueFistMin = np.min(normalizedCycleValues["values"][i][0:int(frameMax),0])
            frameFistMin = np.argmin(normalizedCycleValues["values"][i][0:int(frameMax),0])

            valueSecondMin = np.min(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])
            frameSecondMin = int(frameMax) + np.argmin(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])
 

            label = ["KM1","TKM1"]
            axis = "X"
            desc = ["first max flex moment", "frame of first max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueFistMin,desc[0],
                                               "")
                                               
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],frameFistMin,desc[0],
                                               "")
                                               
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

        # abductor moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(frameMin),1])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(frameMin),1])

            
            values = normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),1]
            indexes = detect_peaks(-values, show=False)

            valueSecondMin = values[indexes[-1]] # take the last index
            frameSecondMin = int(frameMax) + indexes[-1]
 

            label = ["KM4","TKM4"]
            axis = "Y"
            desc = ["max abd moment", "frame of max abdu moment"]
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
            
            
            label = ["KM5","TKM5"]
            axis = "Y"
            desc = ["first max add moment", "frame of first max add moment"]
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

            label = ["KM6","TKM6"]
            axis = "Y"
            desc = ["second max add moment", "frame of second max add moment"]
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

        # extrenal moment
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
            
    def __getAnkle_kinetics(self,analysisInstance,pointLabel,context):
        
        
        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()        
        
        # extensor moment

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
                                               
        # eversor moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])


            label = ["AM3","TAM3"]
            axis = "Y"
            desc = [" max ever moment", "frame of max eve moment"]
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
            desc = [" max inv moment", "frame of max in moment"]
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