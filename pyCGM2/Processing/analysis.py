# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:52:09 2016

@author: fabien Leboeuf ( Salford Univ)
"""
import pdb
import numpy as np
import pandas as pd
import logging

# pyCGM2

import pyCGM2.Processing.cycle as CGM2cycle
import pyCGM2.Tools.exportTools as CGM2exportTools

# openMA

import ma.io
import ma.body


#---- MODULE METHODS ------

class staticAnalysisFilter(object):
    def __init__(self,trial,
                 angleList,
                 subjectInfos=None,
                 modelInfos= None,
                 experimentalInfos=None):
        
        self.m_trial =  trial             
        self.m_angles = angleList
        self.m_subjectInfos = subjectInfos
        self.m_modelInfos = modelInfos
        self.m_experimentalInfos = experimentalInfos

    def buildDataFrame(self):
        df_collection=[]    
        
        for angle in self.m_angles:
            df=pd.DataFrame({"Mean" :self.m_trial.findChild(ma.T_TimeSequence, angle).data().mean(axis=0)[0:3].T})
            df['Axe']=['X','Y','Z']
            df['Label']=angle
            
            if angle[0] == "L":
                df['Side'] = "Left" 
            elif angle[0] == "R":
                df['Side'] = "Right"
            else:
                df['Side'] = "NA"

                
            if self.m_subjectInfos !=None:         
                for key,value in self.m_subjectInfos.items():
                    df[key] = value
            
            if self.m_modelInfos !=None:         
                for key,value in self.m_modelInfos.items():
                    df[key] = value
    
            if self.m_experimentalInfos !=None:         
                for key,value in self.m_experimentalInfos.items():
                    df[key] = value                
    
            df_collection.append(df)
            
            self.m_dataframe = pd.concat(df_collection,ignore_index=True)

    def exportDataFrame(self,outputName,path=None):
        if hasattr(self, 'm_dataframe'):
            if path == None:
                self.m_dataframe.to_csv(str(outputName + " - DataFrame.csv"),sep=";")
            else:
                self.m_dataframe.to_csv(str(path+"/"+outputName + " - DataFrame.csv"),sep=";")
        else:
            raise Exception ("[pyCGM2] - You need to build dataframe before export => RUN buildDataFrame() of your instance")
    
        






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
        pstOut = self.__concreteAnalysisBuilder.computeSpatioTemporel()
        self.analysis.setStp(pstOut)   
        
        kinematicOut,matchPst_kinematic = self.__concreteAnalysisBuilder.computeKinematics()
        self.analysis.setKinematic(kinematicOut, pst= matchPst_kinematic)

        kineticOut,matchPst_kinetic = self.__concreteAnalysisBuilder.computeKinetics()
        self.analysis.setKinetic(kineticOut, pst= matchPst_kinetic)

        if self.__concreteAnalysisBuilder.m_emgs :
            emgOut,matchPst_emg = self.__concreteAnalysisBuilder.computeEmgEnvelopes()
            self.analysis.setEmg(emgOut, pst = matchPst_emg)


    def exportBasicDataFrame(self,outputName, path=None,excelFormat = "xls"):
        if path == None:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- basic.xlsx"))
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(path+"/"+outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
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
        else:
            serie_subject = pd.Series()
        
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
            logging.info("basic dataFrame [%s- basic] Exported"%outputName)





    def exportAdvancedDataFrame(self,outputName, path=None, excelFormat = "xls",csvFileExport =False):
        """
        
        """ 
        if path == None:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- Advanced.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- Advanced.xlsx"))
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(path+"/"+outputName + "- Advanced.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
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

        logging.info("advanced dataFrame [%s- Advanced] Exported"%outputName)             

        xlsxWriter.save()

    def exportAnalysisC3d(self,outputName, path=None):
        
        root = ma.Node('root')    
        trial = ma.Trial("AnalysisC3d",root)
        
        # metadata
        #-------------        
        
        # subject infos
        if self.__concreteAnalysisBuilder.m_subjectInfos != None:          
            subjInfo =  self.__concreteAnalysisBuilder.m_subjectInfos
            for item in subjInfo.items():
                trial.setProperty("SUBJECT_INFO:"+str(item[0]),item[1])
            
        # model infos
        if self.__concreteAnalysisBuilder.m_modelInfos != None:          
            modelInfo =  self.__concreteAnalysisBuilder.m_modelInfos
            for item in modelInfo.items():
                trial.setProperty("MODEL_INFO:"+str(item[0]),item[1])        

        # model infos
        if self.__concreteAnalysisBuilder.m_experimentalConditionInfos != None:          
            experimentalConditionInfo =  self.__concreteAnalysisBuilder.m_experimentalConditionInfos
            for item in experimentalConditionInfo.items():
                trial.setProperty("EXPERIMENTAL_INFO:"+str(item[0]),item[1]) 


        #trial.setProperty('MY_GROUP:MY_PARAMETER',10.0)
        
        # kinematic cycles
        #------------------

        # metadata
        for key in self.analysis.kinematicStats.data.keys():
            if key[1]=="Left":
                n_left_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
                trial.setProperty('PROCESSING:LeftKinematicCycleNumber',n_left_cycle)
                break
        
        for key in self.analysis.kinematicStats.data.keys():
            if key[1]=="Right":
                n_right_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
                trial.setProperty('PROCESSING:RightKinematicCycleNumber',n_right_cycle)
                break
        
        # cycles
        for key in self.analysis.kinematicStats.data.keys():
            label = key[0]
            context = key[1]
            cycle = 0
            values = np.zeros((101,4))            
            for val in self.analysis.kinematicStats.data[label,context]["values"]:
                angle = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),4,101,1.0,0.0,ma.TimeSequence.Type_Angle,"deg", trial.timeSequences())
                values[:,0:3] = val                
                angle.setData(values)
                cycle+=1
                
        # kinetic cycles
        #------------------

        # metadata
        for key in self.analysis.kineticStats.data.keys():
            if key[1]=="Left":
                n_left_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
                trial.setProperty('PROCESSING:LeftKineticCycleNumber',n_left_cycle)
                break
        
        for key in self.analysis.kineticStats.data.keys():
            if key[1]=="Right":
                n_right_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
                trial.setProperty('PROCESSING:RightKineticCycleNumber',n_right_cycle)
                break
        
        # cycles
        for key in self.analysis.kineticStats.data.keys():
            label = key[0]
            context = key[1]
            cycle = 0
            values = np.zeros((101,4))            
            for val in self.analysis.kineticStats.data[label,context]["values"]:
                moment = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),4,101,1.0,0.0,ma.TimeSequence.Type_Moment,"N.mm", trial.timeSequences())
                values[:,0:3] = val                
                moment.setData(values)
                cycle+=1         
        

        try:                
            if path == None:
                ma.io.write(root,str(outputName+".c3d"))
            else:
                ma.io.write(root,str(path + outputName+".c3d"))
            logging.info("Analysis c3d  [%s.c3d] Exported" %( str(outputName +".c3d")) )
        except:
            raise Exception ("[pyCGM2] : analysis c3d doesn t export" )            
    

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

        logging.info("--stp computation--")
        if self.m_cycles.spatioTemporalCycles is not None :

            enableLeftComputation = len ([cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context=="Left"])
            enableRightComputation = len ([cycle for cycle in self.m_cycles.spatioTemporalCycles if cycle.enableFlag and cycle.context=="Right"])

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                if enableLeftComputation:
                    out[label,"Left"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.spatioTemporalCycles,label,"Left")

                if enableRightComputation:
                    out[label,"Right"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.spatioTemporalCycles,label,"Right")
            if enableLeftComputation:        
                logging.info("left stp computation---> done")
            if enableRightComputation:                
                logging.info("right stp computation---> done")
        else:
            logging.warning("No spatioTemporal computation")
        
        return out
        
    def computeKinematics(self):
        """
        calcul toutes les valeurs Moyennes pour les labels de points 
        """

        out={}
        outPst={}

        logging.info("--kinematic computation--")
        if self.m_cycles.kinematicCycles is not None:
            if "Left" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Left"]:
                    labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label 
                    out[labelPlus,"Left"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kinematicCycles,labelPlus,"Left")

                for label in CGM2cycle.GaitCycle.STP_LABELS:
                    outPst[label,"Left"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.kinematicCycles,label,"Left")
                
                logging.info("left kinematic computation---> done")
            else:
                logging.warning("No left Kinematic computation")

            if "Right" in self.m_kinematicLabelsDict.keys():
                for label in self.m_kinematicLabelsDict["Right"]:
                    labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                    out[labelPlus,"Right"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kinematicCycles,labelPlus,"Right")

                for label in CGM2cycle.GaitCycle.STP_LABELS:                    
                    outPst[label,"Right"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.kinematicCycles,label,"Right")
                
                logging.info("right kinematic computation---> done")
            else:
                logging.warning("No right Kinematic computation")

        else:
            logging.warning("No Kinematic computation")

        return out,outPst
        
        
    def computeKinetics(self):
        """
         
        """
        out={}
        outPst={}

        logging.info("--kinetic computation--")
        if self.m_cycles.kineticCycles is not None:
            
           found_context = list() 
           for cycle in self.m_cycles.kineticCycles:
               found_context.append(cycle.context)
           

           if "Left" in self.m_kineticLabelsDict.keys():
               if "Left" in found_context:
                   for label in self.m_kineticLabelsDict["Left"]:
                       labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                       out[labelPlus,"Left"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Left")
                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                        outPst[label,"Left"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.kineticCycles,label,"Left")
                   logging.info("left kinetic computation---> done")
               else:
                   logging.warning("No left Kinetic computation")

                    
           if "Right" in self.m_kineticLabelsDict.keys(): 
               if  "Right" in found_context:                
                   for label in self.m_kineticLabelsDict["Right"]:
                       labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label
                       out[labelPlus,"Right"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Right")
                            
                   for label in CGM2cycle.GaitCycle.STP_LABELS:
                        outPst[label,"Right"]=CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.kineticCycles,label,"Right")
                    
                   logging.info("right kinetic computation---> done")
               else:
                   logging.warning("No right Kinetic computation")

        else:
            logging.warning("No Kinetic computation")

        return out,outPst   
        
    def computeEmgEnvelopes(self):

        out={}
        outPst={}

        logging.info("--emg computation--")        
        if self.m_cycles.emgCycles is not None:

            for rawLabel,muscleDict in zip(self.m_emgLabelList,self.m_emgs):
                                
                muscleLabel = muscleDict["label"]
                muscleSide = muscleDict["side"]

                out[muscleLabel,muscleSide,"Left"]=CGM2cycle.analog_descriptiveStats(self.m_cycles.emgCycles,rawLabel,"Left")
                out[muscleLabel,muscleSide,"Right"]=CGM2cycle.analog_descriptiveStats(self.m_cycles.emgCycles,rawLabel,"Right")

            for label in CGM2cycle.GaitCycle.STP_LABELS:
                outPst[label,"Left"]= CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.emgCycles,label,"Left")
                outPst[label,"Right"]= CGM2cycle.spatioTemporelParameter_descriptiveStats(self.m_cycles.emgCycles,label,"Right")

        else:
            logging.warning("No emg computation")

        return out,outPst