# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd
import logging

# pyCGM2

import pyCGM2.Processing.cycle as CGM2cycle
from pyCGM2.Tools import exportTools 

# openMA

import ma.io
import ma.body


# ---- PATTERN BUILDER ------

# --- OBJECT TO BUILD-----

class StaticAnalysis(): 
    """
        Object built from StaticAnalysisFilter.build(). Its member *data* is a pandas dataframe  
        
    """

    def __init__(self):

        self.data=None


class Analysis(): 
    """
       Object built from AnalysisFilter.build(). 
       
       Analysis work as **class-container**. Its attribute members return descriptive statistics    
       
       Attributes :
      
          - `stpStats` (dict)  - descritive statictics of stapiotemporal parameters         
          - `kinematicStats` (Analysis.Structure)  - descritive statictics of kinematics data.          
          - `kineticStats` (Analysis.Structure)  - descritive statictics of kinetics data.          
          - `emgStats` (Analysis.Structure)  - descritive statictics of emg data.          

       .. note: 
           
           Notice kinematicStats, kineticStats and emgStats are `Analysis.Structure object`. This object implies two dictionnary as sublevels.
            
             - `data` collect descriptive statistics of either kinematics, kinetics or emg. 
             - `pst` returns the spatiotemporal parameters of all cycles involved in either kinematics, kinetics or emg processing.
             
        
    """

    class Structure:
        data = dict()
        pst = dict()
        optionalData = dict()
   
    def __init__(self):

        self.stpStats=dict()
        self.kinematicStats = Analysis.Structure()
        self.kineticStats=Analysis.Structure()
        self.emgStats=Analysis.Structure()
        self.gps= None
        self.gvs = None
        self.coactivations=[]

    def setStp(self,inDict):   
        self.stpStats = inDict    
      
    def setKinematic(self,data, pst = dict()):   
        self.kinematicStats.data = data
        self.kinematicStats.pst = pst 

    def setKinetic(self,data, pst = dict(), optionalData=dict()):   
        self.kineticStats.data = data
        self.kineticStats.pst = pst
        self.kineticStats.optionalData = optionalData 


    def setEmg(self,data, pst = dict()):   
        self.emgStats.data = data
        self.emgStats.pst = pst 

    def setGps(self,GpsStatsOverall,GpsStatsContext):
        self.gps = dict()
        self.gps["Overall"] = GpsStatsOverall
        self.gps["Context"] = GpsStatsContext

    def setGvs(self,gvsStats):
        self.gvs = gvsStats


    def getKinematicCycleNumbers(self):
        for label,context in self.kinematicStats.data.keys():
            if context == "Left":
                n_leftCycles = self.kinematicStats.data[label, context]["values"].__len__()
                break

        for label,context in self.kinematicStats.data.keys():
            if context == "Right":
                n_rightCycles = self.kinematicStats.data[label, context]["values"].__len__()
                break
        return n_leftCycles, n_rightCycles
        
# --- BUILDERS-----
class AbstractBuilder(object):
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
    """
        **Description** : builder of a common clinical gait analysis         
    """
    
    
    
    def __init__(self,cycles,
                 kinematicLabelsDict=None ,
                 kineticLabelsDict =None,
                 pointlabelSuffix = "",
                 emgLabelList = None, 
                 modelInfos = None, subjectInfos = None, experimentalInfos = None, emgs = None):

        """
            :Parameters:
                 - `cycles` (pyCGM2.Processing.cycle.Cycles) - Cycles instance built from CycleFilter
                 - `kinematicLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinematic output label    
                 - `kineticLabelsDict` (dict) - dictionnary with two items (Left and Right) grouping kinetic output label   
                 - `pointlabelSuffix` (dict) - suffix ending kinematicLabels and kineticLabels dictionnaries
                 - `emgLabelList` (list of str) - labels of used emg   
                 - `subjectInfos` (dict) - information about the subject   
                 - `modelInfos` (dict) - information about the model   
                 - `experimentalInfos` (dict) - information about the experimental conditions
                 -  .. attention:: `emgs` (pyCGM2emg) - object in progress 


            .. note::
            
                modelInfos,experimentalInfos, subjectInfos are convenient dictionnaries in which you can store different sort of information


                

            

        """


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

        """ 
            **Description:** compute descriptive of spatio-temporal parameters 

            :return:
                - `out` (dict) - dictionnary with descriptive statictics of spatio-temporal parameters
                
        """ 
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
        """ compute descriptive of kinematics parameters 
        
            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinematics parameters 
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinematics parameters
        
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
        """ compute descriptive of kinetics parameters 
        
            :return:
                - `out` (dict) - dictionnary with descriptive statictics of kinetics parameters 
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching  kinetics parameters
        
        """ 


        out={}
        outPst={}
        outOptional={}

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
                   for label in self.m_kinematicLabelsDict["Left"]:
                       labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label 
                       outOptional[labelPlus,"Left"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Left")
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

                   for label in self.m_kinematicLabelsDict["Right"]:
                       labelPlus = label + "_" + self.m_pointlabelSuffix if self.m_pointlabelSuffix!="" else label 
                       outOptional[labelPlus,"Right"]=CGM2cycle.point_descriptiveStats(self.m_cycles.kineticCycles,labelPlus,"Right")

                    
                   logging.info("right kinetic computation---> done")
               else:
                   logging.warning("No right Kinetic computation")


        else:
            logging.warning("No Kinetic computation")

        return out,outPst,outOptional   
        
    def computeEmgEnvelopes(self):
        
        """ 
            Compute descriptive of emg values
            
            :return:
                - `out` (dict) - dictionnary with descriptive statictics of emg envelopes 
                - `outPst` ( dict) - dictionnary with descriptive statictics of spatio-temporal parameters matching emg envelopes
        """ 


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

# ---- FILTERS ------
class StaticAnalysisFilter(object):
    """
        filter building a StaticAnalysis instance from a static trial        
        
    """
    
    def __init__(self,trial,
                 angleList,
                 subjectInfos=None,
                 modelInfos= None,
                 experimentalInfos=None):

        """
            :Parameters:
                 - `trial` (openma-trial) - openma trial from a c3d
                 - `angleList` (list of str) - list of angle labels.    
                 - `subjectInfos` (dict) - information about the subject   
                 - `modelInfos` (dict) - information about the model   
                 - `experimentalInfos` (dict) - information about the experimental conditions   

        """

        self.analysis = StaticAnalysis()

        
        self.m_trial =  trial             
        self.m_angles = angleList
        self.m_subjectInfos = subjectInfos
        self.m_modelInfos = modelInfos
        self.m_experimentalInfos = experimentalInfos



    def build(self):
        
        """
            build a StaticAnalysis instance  
    
            .. warning::  Last frame is removed from mean calculation because the IK engine of opensim can return an error on the last frame 
        """        
        
        df_collection=[]    
        
        for angle in self.m_angles:
            
            data = self.m_trial.findChild(ma.T_TimeSequence, angle).data()
            if data.shape[0]<=2:
                logging.warning("Frame inferior to 2 in your static file. Get the first frame only")
                df=pd.DataFrame({"Mean" :self.m_trial.findChild(ma.T_TimeSequence, angle).data()[0,0:3]}) 
            else:
                df=pd.DataFrame({"Mean" :self.m_trial.findChild(ma.T_TimeSequence, angle).data()[:-1].mean(axis=0)[0:3].T})

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
            
            self.analysis.data = pd.concat(df_collection,ignore_index=True)

    def exportDataFrame(self,outputName,path=None):
        """
             export the member *analysis* as xls file 
    
            :Parameters:
                - `outputName` (str) - name of the xls file ( without xls extension)
                - `path` (str) - folder in which xls files will be stored
                
        """        
        
        if path == None:
            xlsxWriter = pd.ExcelWriter(str(outputName + "- static.xls"),engine='xlwt')
            self.analysis.data.to_excel(xlsxWriter,'mean static')
        else:
            xlsxWriter = pd.ExcelWriter(str(path+"/"+outputName + "- static.xls"),engine='xlwt')
            self.analysis.data.to_excel(xlsxWriter,'mean static')
        
        xlsxWriter.save()

    
        
class AnalysisFilter(object):
    """
         Filter building an Analysis instance. 
    """            

    def __init__(self):
        self.__concreteAnalysisBuilder = None
        self.analysis = Analysis() 
    
    def setBuilder(self,concreteBuilder):
        """
             set a concrete builder
    
            :Parameters:
                - `concreteBuilder` (Builder) - a concrete Builder
               
        """          
        
        self.__concreteAnalysisBuilder = concreteBuilder
   
    def build (self) :
        """
            build member analysis from a concrete builder 
        
        """
        pstOut = self.__concreteAnalysisBuilder.computeSpatioTemporel()
        self.analysis.setStp(pstOut)   
        
        kinematicOut,matchPst_kinematic = self.__concreteAnalysisBuilder.computeKinematics()
        self.analysis.setKinematic(kinematicOut, pst= matchPst_kinematic)

        kineticOut,matchPst_kinetic,matchKinematic = self.__concreteAnalysisBuilder.computeKinetics()
        self.analysis.setKinetic(kineticOut, pst= matchPst_kinetic, optionalData=matchKinematic)

        if self.__concreteAnalysisBuilder.m_emgs :
            emgOut,matchPst_emg = self.__concreteAnalysisBuilder.computeEmgEnvelopes()
            self.analysis.setEmg(emgOut, pst = matchPst_emg)


    

#    def exportAnalysisC3d(self,outputName, path=None):
#        """
#             export a  101-frames c3d grouping all cycle parameters  
#    
#            :Parameters:
#                - `outputName` (str) - name of the xls file ( without xls extension)
#                - `path` (str) - folder in which xls files will be stored
#                
#        """
#
#        
#        root = ma.Node('root')    
#        trial = ma.Trial("AnalysisC3d",root)
#        
#        # metadata
#        #-------------        
#        
#        # subject infos
#        if self.__concreteAnalysisBuilder.m_subjectInfos != None:          
#            subjInfo =  self.__concreteAnalysisBuilder.m_subjectInfos
#            for item in subjInfo.items():
#                trial.setProperty("SUBJECT_INFO:"+str(item[0]),item[1])
#            
#        # model infos
#        if self.__concreteAnalysisBuilder.m_modelInfos != None:          
#            modelInfo =  self.__concreteAnalysisBuilder.m_modelInfos
#            for item in modelInfo.items():
#                trial.setProperty("MODEL_INFO:"+str(item[0]),item[1])        
#
#        # model infos
#        if self.__concreteAnalysisBuilder.m_experimentalConditionInfos != None:          
#            experimentalConditionInfo =  self.__concreteAnalysisBuilder.m_experimentalConditionInfos
#            for item in experimentalConditionInfo.items():
#                trial.setProperty("EXPERIMENTAL_INFO:"+str(item[0]),item[1]) 
#
#
#        #trial.setProperty('MY_GROUP:MY_PARAMETER',10.0)
#        
#        # kinematic cycles
#        #------------------
#
#        # metadata
#        for key in self.analysis.kinematicStats.data.keys():
#            if key[1]=="Left":
#                n_left_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
#                trial.setProperty('PROCESSING:LeftKinematicCycleNumber',n_left_cycle)
#                break
#        
#        for key in self.analysis.kinematicStats.data.keys():
#            if key[1]=="Right":
#                n_right_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
#                trial.setProperty('PROCESSING:RightKinematicCycleNumber',n_right_cycle)
#                break
#        
#        # cycles
#        for key in self.analysis.kinematicStats.data.keys():
#            label = key[0]
#            context = key[1]
#            cycle = 0
#            values = np.zeros((101,4))            
#            for val in self.analysis.kinematicStats.data[label,context]["values"]:
#                angle = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),4,101,1.0,0.0,ma.TimeSequence.Type_Angle,"deg", trial.timeSequences())
#                values[:,0:3] = val                
#                angle.setData(values)
#                cycle+=1
#                
#        # kinetic cycles
#        #------------------
#
#        # metadata
#        for key in self.analysis.kineticStats.data.keys():
#            if key[1]=="Left":
#                n_left_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
#                trial.setProperty('PROCESSING:LeftKineticCycleNumber',n_left_cycle)
#                break
#        
#        for key in self.analysis.kineticStats.data.keys():
#            if key[1]=="Right":
#                n_right_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
#                trial.setProperty('PROCESSING:RightKineticCycleNumber',n_right_cycle)
#                break
#        
#        # cycles
#        for key in self.analysis.kineticStats.data.keys():
#            label = key[0]
#            context = key[1]
#            cycle = 0
#            values = np.zeros((101,4))            
#            for val in self.analysis.kineticStats.data[label,context]["values"]:
#                moment = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),4,101,1.0,0.0,ma.TimeSequence.Type_Moment,"N.mm", trial.timeSequences())
#                values[:,0:3] = val                
#                moment.setData(values)
#                cycle+=1         
#        
#
#        try:                
#            if path == None:
#                ma.io.write(root,str(outputName+".c3d"))
#            else:
#                ma.io.write(root,str(path + outputName+".c3d"))
#            logging.info("Analysis c3d  [%s.c3d] Exported" %( str(outputName +".c3d")) )
#        except:
#            raise Exception ("[pyCGM2] : analysis c3d doesn t export" )            
    
