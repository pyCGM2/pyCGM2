# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:52:09 2016

@author: fabien Leboeuf ( Salford Univ)
"""
import pdb
import numpy as np
import pandas as pd

# openMA
import ma.io
import ma.body

# pyCGM2
import cycle as CGM2cycle
import pyCGM2.Core.Tools.exportTools as CGM2exportTools


def spatioTemporelParameter_descriptiveStats(cycles,label,context):
    
    """   
    """

    outDict=dict()    
    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    val = np.zeros((n))
    
    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            val[i] = cycle.getSpatioTemporalParameter(label)
            i+=1
    outDict = {'mean':np.mean(val),'std':np.std(val),'median':np.median(val),'values': val}
            
    return outDict


def point_descriptiveStats(cycles,label,context):
    """
    """
       
    outDict=dict()    
    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    
    x=np.empty((101,n))
    y=np.empty((101,n))
    z=np.empty((101,n))

    listOfPointValues=list()

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            tmp = cycle.getPointTimeSequenceDataNormalized(label)
            x[:,i]=tmp[:,0]
            y[:,i]=tmp[:,1]
            z[:,i]=tmp[:,2]
            
            listOfPointValues.append(tmp)
            
            i+=1
                
            
    meanData=np.array(np.zeros((101,3)))    
    meanData[:,0]=np.mean(x,axis=1)
    meanData[:,1]=np.mean(y,axis=1)
    meanData[:,2]=np.mean(z,axis=1)
    
    stdData=np.array(np.zeros((101,3)))    
    stdData[:,0]=np.std(x,axis=1)
    stdData[:,1]=np.std(y,axis=1)
    stdData[:,2]=np.std(z,axis=1)


    medianData=np.array(np.zeros((101,3)))    
    medianData[:,0]=np.median(x,axis=1)
    medianData[:,1]=np.median(y,axis=1)
    medianData[:,2]=np.median(z,axis=1)


    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': listOfPointValues }
    

            
    return outDict
    
    
    
def analog_descriptiveStats(cycles,label,context):

    outDict=dict()    


    
    n=len([cycle for cycle in cycles if cycle.enableFlag and cycle.context==context]) # list comprehension , get number of enabled cycle 
    
    x=np.empty((1001,n))

    i=0
    for cycle in cycles:
        if cycle.enableFlag and cycle.context==context:
            tmp = cycle.getAnalogTimeSequenceDataNormalized(label)
            x[:,i]=tmp[:,0]
          
            
            i+=1

    x_resize=x[0:1001:10,:]
            
    meanData=np.array(np.zeros((101,1)))    
    meanData[:,0]=np.mean(x_resize,axis=1)
    
    stdData=np.array(np.zeros((101,1)))    
    stdData[:,0]=np.std(x_resize,axis=1)

    medianData=np.array(np.zeros((101,1)))    
    medianData[:,0]=np.median(x_resize,axis=1)


    outDict = {'mean':meanData, 'median':medianData, 'std':stdData, 'values': x_resize }
            
    return outDict



# ---- FILTER ------
class AnalysisFilter(object): # CONTROLER

    def __init__(self):
        self.__concreteAnalysisBuilder = None
        self.analysis = Analysis() # output stats container 
    
    def setBuilder(self,concreteBuilder):
        self.__concreteAnalysisBuilder = concreteBuilder
   
    def build (self) :
        
        """
        
        """
        print "####### ANALYSIS FILTER #######"
        pstOut = self.__concreteAnalysisBuilder.computeSpatioTemporel()
        self.analysis.setStp(pstOut)   
        
        kinematicOut,matchPst_kinematic = self.__concreteAnalysisBuilder.computeKinematics()
        self.analysis.setKinematic(kinematicOut, pst= matchPst_kinematic)

        kineticOut,matchPst_kinetic = self.__concreteAnalysisBuilder.computeKinetics()
        self.analysis.setKinetic(kineticOut, pst= matchPst_kinetic)

        if self.__concreteAnalysisBuilder.m_emgs :
            emgOut,matchPst_emg = self.__concreteAnalysisBuilder.computeEmgEnvelopes()
            self.analysis.setEmg(emgOut, pst = matchPst_emg)


    def exportBasicDataFrame(self,outputName, path=None):
        if path == None:
            xlsxWriter = pd.ExcelWriter(str(outputName + "- basic.xlsx"))
        else:
            xlsxWriter = pd.ExcelWriter(str(path+"/"+outputName + "- basic.xlsx"))


        # metadata
        #--------------
        if self.__concreteAnalysisBuilder.m_subjectInfos != None:          
            subjInfo =  self.__concreteAnalysisBuilder.m_subjectInfos
        else:
            subjInfo=None
        
        if self.__concreteAnalysisBuilder.m_modelInfos != None:          
            modelInfo =  self.__concreteAnalysisBuilder.m_modelInfos
        else:
            modelInfo=None
            
        if self.__concreteAnalysisBuilder.m_experimentalConditionInfos != None:          
            experimentalConditionInfo =  self.__concreteAnalysisBuilder.m_experimentalConditionInfos
        else:
            experimentalConditionInfo=None

        list_index =list()# use for sorting index
        if subjInfo !=None:
            for key in subjInfo:
                list_index.append(key)
            serie_subject = pd.Series(subjInfo)
        
        if modelInfo !=None:
            for key in modelInfo:
                list_index.append(key)
            serie_model = pd.Series(modelInfo)
        else:
            serie_model = pd.Series()
            
        if experimentalConditionInfo !=None:
            for key in experimentalConditionInfo:
                list_index.append(key)
            serie_exp = pd.Series(experimentalConditionInfo)
        else:
            serie_exp = pd.Series()

        df_metadata = pd.DataFrame({"subjectInfos": serie_subject,
                                    "modelInfos": serie_model,
                                    "experimentInfos" : serie_exp},
                                    index = list_index)


        df_metadata.to_excel(xlsxWriter,"Infos")

    


        if self.analysis.kinematicStats.data!={}:

            # spatio temporal paramaters matching Kinematic cycles
            dfs_l =[]
            dfs_r =[]
            for key in self.analysis.kinematicStats.pst.keys():
                label = key[0]
                context = key[1]
                n =len(self.analysis.kinematicStats.pst[label,context]['values'])
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                
                if context == "Left":
                    df = pd.DataFrame.from_items([(label, self.analysis.kinematicStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_l.append(df)

                if context == "Right":
                    df = pd.DataFrame.from_items([(label, self.analysis.kinematicStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_r.append(df)

            if dfs_l !=[]:
                df_pst_L=pd.concat(dfs_l)  
                df_pst_L.to_excel(xlsxWriter,"Left - stp-kinematics")                

            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r) 
                df_pst_R.to_excel(xlsxWriter,"Right - stp-kinematics")


            # kinematic cycles
            for key in self.analysis.kinematicStats.data.keys():
                label = key[0]
                context = key[1]
                n = len(self.analysis.kinematicStats.data[label,context]["values"])
                X = np.zeros((101,n))                
                Y = np.zeros((101,n)) 
                Z = np.zeros((101,n)) 
                for i in range(0,n):
                    X[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,0]
                    Y[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,1]
                    Z[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,2]
                  
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                frame_header= ["Frame "+str(i) for i in range(0,101)]


                df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                df_x['Axis']='X'
                df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                df_y['Axis']='Y'
                df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                df_z['Axis']='Z'
                
                df_label = pd.concat([df_x,df_y,df_z])
                df_label.to_excel(xlsxWriter,str(label+"."+context)) 
                
        if self.analysis.kineticStats.data!={}:
            # spatio temporal paramaters matching Kinetic cycles
            dfs_l =[]
            dfs_r =[]
            for key in self.analysis.kineticStats.pst.keys():
                label = key[0]
                context = key[1]
                n =len(self.analysis.kineticStats.pst[label,context]['values'])
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                
                if context == "Left":
                    df = pd.DataFrame.from_items([(label, self.analysis.kineticStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_l.append(df)

                if context == "Right":
                    df = pd.DataFrame.from_items([(label, self.analysis.kineticStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_r.append(df)

            if dfs_l !=[]:    
                df_pst_L=pd.concat(dfs_l)
                df_pst_L.to_excel(xlsxWriter,"Left - pst-kinetics")                
           
            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r)
                df_pst_R.to_excel(xlsxWriter,"Right - pst-kinetics")


            # kinetic cycles
            for key in self.analysis.kineticStats.data.keys():
                label=key[0]
                context=key[1]
                
                n = len(self.analysis.kineticStats.data[label,context]["values"])
                X = np.zeros((101,n))                
                Y = np.zeros((101,n)) 
                Z = np.zeros((101,n)) 
                for i in range(0,n):
                    X[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,0]
                    Y[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,1]
                    Z[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,2]
                  
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                frame_header= ["Frame "+str(i) for i in range(0,101)]

                df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                df_x['Axis']='X'
                df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                df_y['Axis']='Y'
                df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                df_z['Axis']='Z'
                
                df_label = pd.concat([df_x,df_y,df_z])
                df_label.to_excel(xlsxWriter,str(label+"."+context)) 

            xlsxWriter.save()
            print "basic dataFrame [%s] Exported"%outputName 

    def exportAdvancedDataFrame(self,outputName, path=None, csvFileExport =False):
        """
        
        """ 
        if path == None:
            xlsxWriter = pd.ExcelWriter(str(outputName + "- Advanced.xlsx"))
        else:
            xlsxWriter = pd.ExcelWriter(str(path+"/"+outputName + "- Advanced.xlsx"))
        

        # infos
        #-------    
        if self.__concreteAnalysisBuilder.m_modelInfos != None:          
            modelInfo =  self.__concreteAnalysisBuilder.m_modelInfos
        else:
            modelInfo=None

        
        if self.__concreteAnalysisBuilder.m_subjectInfos != None:          
            subjInfo =  self.__concreteAnalysisBuilder.m_subjectInfos
        else:
            subjInfo=None

        if self.__concreteAnalysisBuilder.m_experimentalConditionInfos != None:          
            condExpInfo =  self.__concreteAnalysisBuilder.m_experimentalConditionInfos
        else:
            condExpInfo=None

        # spatio temporal parameters
        #---------------------------
                
        if self.analysis.stpStats != {}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveStp = CGM2exportTools.buid_df_descriptiveCycle1_1(self.analysis.stpStats)
            
            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value            
            
            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value
            df_descriptiveStp.to_excel(xlsxWriter,'descriptive stp')
            

            # stage 2 : get cycle values
            # --------------------------------
            df_stp = CGM2exportTools.buid_df_cycles1_1(self.analysis.stpStats)
            
            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_stp, key)
                    df_stp[key] = value 
            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_stp, key)
                    df_stp[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_stp, key)
                    df_stp[key] = value
                        
            df_stp.to_excel(xlsxWriter,'stp cycles')

            if csvFileExport:
                if path == None:
                    df_stp.to_csv(str(outputName + " - stp - DataFrame.csv"),sep=";")
                else:
                    df_stp.to_csv(str(path+"/"+outputName + " - stp - DataFrame.csv"),sep=";")

        
        # Kinematics ouput
        #---------------------------

        
        if self.analysis.kinematicStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------            
            df_descriptiveKinematics = CGM2exportTools.buid_df_descriptiveCycle101_3(self.analysis.kinematicStats)

            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value 
            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value

            df_descriptiveKinematics.to_excel(xlsxWriter,'descriptive kinematics ')                

            # stage 2 : get cycle values
            # --------------------------------
            
            # cycles            
            df_kinematics =  CGM2exportTools.buid_df_cycles101_3(self.analysis.kinematicStats) 

            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value 

            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value
                            
            df_kinematics.to_excel(xlsxWriter,'Kinematic cycles')
            if csvFileExport:
                if path == None:
                    df_kinematics.to_csv(str(outputName + " - stp - DataFrame.csv"),sep=";")
                else:
                    df_kinematics.to_csv(str(path+"/"+outputName + " - stp - DataFrame.csv"),sep=";")
            


        # Kinetic ouputs
        #---------------------------
        if self.analysis.kineticStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------            
            df_descriptiveKinetics = CGM2exportTools.buid_df_descriptiveCycle101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_stp, key)
                    df_descriptiveKinetics[key] = value 
            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value

            df_descriptiveKinetics.to_excel(xlsxWriter,'descriptive kinetics ')                

            # stage 2 : get cycle values
            # --------------------------------
            
            # cycles            
            df_kinetics =  CGM2exportTools.buid_df_cycles101_3(self.analysis.kineticStats) 

            # add infos
            if modelInfo !=None:         
                for key,value in modelInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value 

            if subjInfo !=None:         
                for key,value in subjInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value
            if condExpInfo !=None:         
                for key,value in condExpInfo.items():
                    CGM2exportTools.isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value
                            
            df_kinetics.to_excel(xlsxWriter,'Kinetic cycles')
            if csvFileExport:
                if path == None:
                    df_stp.to_csv(str(outputName + " - stp - DataFrame.csv"),sep=";")
                else:
                    df_stp.to_csv(str(path+"/"+outputName + " - stp - DataFrame.csv"),sep=";")

        print "advanced dataFrame [%s] Exported"%outputName             



# ---- PATTERN BUILDER ------

# --- Object to build-----
class Analysis(): # OBJECT TO CONSTRUCT
    """
   
    """
    class Structure:
        data = dict()
        pst = dict()  
   
    def __init__(self):
        self.stpStats=dict()
        self.kinematicStats = Analysis.Structure()
        self.kineticStats=Analysis.Structure()
        self.emgStats=Analysis.Structure()
        self.gps= None
        self.coactivations=[]

    def setStp(self,inDict):   
        self.stpStats = inDict    
      
    def setKinematic(self,data, pst = dict()):   
        self.kinematicStats.data = data
        self.kinematicStats.pst = pst 

    def setKinetic(self,data, pst = dict()):   
        self.kineticStats.data = data
        self.kineticStats.pst = pst 


    def setEmg(self,data, pst = dict()):   
        self.emgStats.data = data
        self.emgStats.pst = pst 



# --- Builder-----
class AbstractBuilder(object):
    """
    Abstract Builder
    """
    def __init__(self,cycles=None):
        self.m_cycles =cycles        

    def computeSpatioTemporel(self):
        pass

    def computeKinematics(self):
        pass

    def computeKinetics(self,momentContribution = False):
        pass

    def computeEmgEnvelopes(self):
        pass


class GaitAnalysisBuilder(AbstractBuilder):
    def __init__(self,cycles,
                 kinematicLabelsDict=None ,
                 kineticLabelsDict =None,
                 pointlabelSuffix = "",
                 emgLabelList = None, 
                 modelInfos = None, subjectInfos = None, experimentalInfos = None, emgs = None):
        super(GaitAnalysisBuilder, self).__init__(cycles=cycles)

        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_emgLabelList = emgLabelList
        self.m_emgs = emgs
        
        self.m_modelInfos=modelInfos        
        self.m_subjectInfos = subjectInfos
        self.m_experimentalConditionInfos = experimentalInfos
        
        
    def computeSpatioTemporel(self):

        out={}

        if self.m_cycles.spatioTemporalCycles is not None :
            print "spatioTemporal computation"

            enableLeftComputation = len ([cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context=="Left"])
            enableRightComputation = len ([cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context=="Right"])

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                if enableLeftComputation:
                    out[label,"Left"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.spatioTemporalCycles,label,"Left")

                if enableRightComputation:
                    out[label,"Right"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.spatioTemporalCycles,label,"Right")
            if enableLeftComputation:        
                print "---> Left done"
            if enableRightComputation:                
                print "---> Right done"                
        else:
            print "No spatioTemporal computation"
        
        return out
        
    def computeKinematics(self):
        """
        calcul toutes les valeurs Moyennes pour les labels de points 
        """

        out={}
        outPst={}

        if self.m_cycles.kinematicCycles is not None:
            print "kinematic computation"
        
            if "Left" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Left"]:
                    labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label 
                    out[labelPlus,"Left"]=point_descriptiveStats(self.m_cycles.kinematicCycles,labelPlus,"Left")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label,"Left"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.kinematicCycles,label,"Left")
                
                print "---> Left done"
            else:
                print "---> Left no output"

            if "Right" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Right"]:
                    labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                    out[labelPlus,"Right"]=point_descriptiveStats(self.m_cycles.kinematicCycles,labelPlus,"Right")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label,"Right"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.kinematicCycles,label,"Right")
                
                print "---> Right done"
            else:
                print "---> Right no output"

        else:
            print "No kinematic cycle computation" 

        return out,outPst
        
        
    def computeKinetics(self):
        """
         
        """
        out={}
        outPst={}

        if self.m_cycles.kineticCycles is not None:
           print "kinetic computation"
        
           if "Left" in self.m_kinematicLabelsDict.keys():
               for label in self.m_kineticLabelsDict["Left"]:
                   labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                   out[labelPlus,"Left"]=point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Left")
               for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label,"Left"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.kineticCycles,label,"Left")
               print "---> Left done"
           else:
               print "---> Left no output"

                    
           if "Right" in self.m_kinematicLabelsDict.keys():                
               for label in self.m_kineticLabelsDict["Right"]:
                   labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                   out[labelPlus,"Right"]=point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Right")
                        
               for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label,"Right"]=spatioTemporelParameter_descriptiveStats(self.m_cycles.kineticCycles,label,"Right")
                
               print "---> Right done"
           else:
               print "---> Right no output"

        else:
            print "No kinetic cycle computation" 

        return out,outPst   
        
    def computeEmgEnvelopes(self):

        out={}
        outPst={}

        
        if self.m_cycles.emgCycles is not None:
            print "emg computation"

            for rawLabel,muscleDict in zip(self.m_emgLabelList,self.m_emgs):
                                
                muscleLabel = muscleDict["label"]
                muscleSide = muscleDict["side"]

                out[muscleLabel,muscleSide,"Left"]=analog_descriptiveStats(self.m_cycles.emgCycles,rawLabel,"Left")
                out[muscleLabel,muscleSide,"Right"]=analog_descriptiveStats(self.m_cycles.emgCycles,rawLabel,"Right")

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                outPst[label,"Left"]= spatioTemporelParameter_descriptiveStats(self.m_cycles.emgCycles,label,"Left")
                outPst[label,"Right"]= spatioTemporelParameter_descriptiveStats(self.m_cycles.emgCycles,label,"Right")

        else:
            print "no emg computation"

        return out,outPst