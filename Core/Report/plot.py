# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:30:42 2016

@author: fabien Leboeuf
"""

import numpy as np
import scipy as sp
import logging

import pdb

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# pyCGM2
import pyCGM2.Core.Processing.analysis as CGM2analysis

def gaitDescriptivePlot(figAxis,analysis_node, 
                    pointLabel_L,contextPointLabel_L, 
                    pointLabel_R, contextPointLabel_R, 
                    axis,
                    title, xlabel="", ylabel="",
                    leftStaticLimit=None,
                    rightStaticLimit=None,
                    leftLimits=None,
                    rightLimits=None):

    # check if [ pointlabel , context ] in keys of analysis_node 
    left_flag = False    
    for key in analysis_node.data.keys():
        if key[0] == pointLabel_L and key[1] == contextPointLabel_L:
            left_flag = True if analysis_node.data[pointLabel_L,contextPointLabel_L]["values"] != [] else False

            
    right_flag = False
    for key in analysis_node.data.keys():
        if key[0] == pointLabel_R and key[1] == contextPointLabel_R:
            right_flag = True if analysis_node.data[pointLabel_R,contextPointLabel_R]["values"] != [] else False    


    # plot        
    if left_flag:
        mean_L=analysis_node.data[pointLabel_L,contextPointLabel_L]["mean"][:,axis]
        std_L=analysis_node.data[pointLabel_L,contextPointLabel_L]["std"][:,axis]  
        figAxis.plot(np.linspace(0,100,101), mean_L, 'r-')
        figAxis.fill_between(np.linspace(0,100,101), mean_L-std_L, mean_L+std_L, facecolor="red", alpha=0.5,linewidth=0)
        
        if analysis_node.pst !={}:
            stance = analysis_node.pst['stancePhase', 'Left']["mean"]
            double1 = analysis_node.pst['doubleStance1', 'Left']["mean"]
            double2 = analysis_node.pst['doubleStance2', 'Left']["mean"]
            figAxis.axvline(stance,color='r',ls='dashed')
            figAxis.axvline(double1,color='r',ls='dotted')
            figAxis.axvline(stance-double2,color='r',ls='dotted')

        if leftLimits is not None:
            for value in leftLimits:
                figAxis.axhline(value,color='r',ls='dashed')
        if leftStaticLimit is not None:
            figAxis.axhline(leftStaticLimit,color='r',ls='dotted')
                
        
        

    if right_flag:
        mean_R=analysis_node.data[pointLabel_R,contextPointLabel_R]["mean"][:,axis]
        std_R=analysis_node.data[pointLabel_R,contextPointLabel_R]["std"][:,axis]
        figAxis.plot(np.linspace(0,100,101), mean_R, 'b-')
        figAxis.fill_between(np.linspace(0,100,101), mean_R-std_R, mean_R+std_R, facecolor="blue", alpha=0.5,linewidth=0)

        if analysis_node.pst !={}:
            stance = analysis_node.pst['stancePhase', 'Right']["mean"]
            double1 = analysis_node.pst['doubleStance1', 'Right']["mean"]
            double2 = analysis_node.pst['doubleStance2', 'Right']["mean"]
            figAxis.axvline(stance,color='b',ls='dashed')
            figAxis.axvline(double1,color='b',ls='dotted')
            figAxis.axvline(stance-double2,color='b',ls='dotted')

        if rightLimits is not None:
            for value in rightLimits:
                figAxis.axhline(value,color='b',ls='dashed')

        if rightStaticLimit is not None:
            figAxis.axhline(rightStaticLimit,color='b',ls='dotted')


    figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    figAxis.set_xlabel(xlabel,size=8) 
    figAxis.set_ylabel(ylabel,size=8)


def gaitConsistencyPlot( figAxis, analysis_node,  pointLabel_L,contextPointLabel_L, pointLabel_R, contextPointLabel_R, axis, title, xlabel="", ylabel="",
                    leftLimits=None,
                    rightLimits=None,
                    leftStaticLimit=None,
                    rightStaticLimit=None):

    # Left plot
    #------------
    # check pointLabel, contextpoint exist
    left_flag = False
    for key in analysis_node.data.keys():
        if key[0] == pointLabel_L and key[1] == contextPointLabel_L:
           n = len(analysis_node.data[pointLabel_L,contextPointLabel_L]["values"])
           left_flag = True if n !=0 else False   
           
    # plot    
    if left_flag:
        values_L= np.zeros((101,n))
        i=0
        for val in analysis_node.data[pointLabel_L,contextPointLabel_L]["values"]:
            values_L[:,i] = val[:,axis]
            i+=1

        figAxis.plot(np.linspace(0,100,101), values_L, 'r-')
        
        if analysis_node.pst !={}:
            for valStance,valDouble1,valDouble2, in zip(analysis_node.pst['stancePhase', 'Left']["values"],analysis_node.pst['doubleStance1', 'Left']["values"],analysis_node.pst['doubleStance2', 'Left']["values"]):     
                figAxis.axvline(valStance,color='r',ls='dashed')
                figAxis.axvline(valDouble1,color='r',ls='dotted')
                figAxis.axvline(valStance-valDouble2,color='r',ls='dotted')
                
                
        if leftLimits is not None:
            for value in leftLimits:
                figAxis.axhline(value,color='r',ls='dashed')
        if leftStaticLimit is not None:
            figAxis.axhline(leftStaticLimit,color='r',ls='dotted')

    # right plot
    #------------
    # check
    right_flag = False
    for key in analysis_node.data.keys():
        if key[0] == pointLabel_R and key[1] == contextPointLabel_R:
            n = len(analysis_node.data[pointLabel_R,contextPointLabel_R]["values"])
            right_flag = True if n !=0 else False

    # plot
    if right_flag:
        values_R= np.zeros((101,n))
        i=0
        for val in analysis_node.data[pointLabel_R,contextPointLabel_R]["values"]:
            values_R[:,i] = val[:,axis]
            i+=1

        figAxis.plot(np.linspace(0,100,101), values_R, 'b-')
        
        if analysis_node.pst !={}:
            for valStance,valDouble1,valDouble2, in zip(analysis_node.pst['stancePhase', 'Right']["values"],analysis_node.pst['doubleStance1', 'Right']["values"],analysis_node.pst['doubleStance2', 'Right']["values"]):     
                figAxis.axvline(valStance,color='b',ls='dashed')
                figAxis.axvline(valDouble1,color='b',ls='dotted')
                figAxis.axvline(valStance-valDouble2,color='b',ls='dotted')
                
        if rightLimits is not None:
            for value in rightLimits:
                figAxis.axhline(value,color='b',ls='dashed')

        if rightStaticLimit is not None:
            figAxis.axhline(rightStaticLimit,color='b',ls='dotted')


    figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100.0])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    figAxis.set_xlabel(xlabel,size=8) 
    figAxis.set_ylabel(ylabel,size=8)


# ------ FILTER -----------
class PlottingFilter(object):
    """  """      
    def __init__(self):
        self.__concretePlotBuilder = None
        self.m_path = None
        self.m_pdfSuffix = None
        self.m_nd_procedure = None

    def setPath(self,path):
        self.m_path = path    
        
    def setNormativeDataProcedure(self,ndp):
        self.m_nd_procedure = ndp    
        
    def setPdfSuffix(self,suffix):
        self.m_pdfSuffix = suffix    
   
    def setBuilder(self,concretePlotBuilder):
        self.__concretePlotBuilder = concretePlotBuilder    
    
    def plot(self, consistencyOnly=True):
        
        if self.m_nd_procedure is None:
            self.__concretePlotBuilder.plotPanel(self.m_path,self.m_pdfSuffix,consistencyOnly=consistencyOnly)

        else:
            self.m_nd_procedure.constructNormativeData()
            self.__concretePlotBuilder.plotPanel(self.m_path,self.m_pdfSuffix,consistencyOnly=consistencyOnly,normativeData = self.m_nd_procedure.data)

# ------ BUILDER -----------
class AbstractPlotBuilder(object):
    """
    Abstract Builder
    """
    def __init__(self,iObj=None):
        self.m_input =iObj

    def plotPanel(self):
        pass


class GaitAnalysisPlotBuilder(AbstractPlotBuilder):
    def __init__(self,iObj,kineticFlag=True,pointLabelSuffix=""):
        super(GaitAnalysisPlotBuilder, self).__init__(iObj=iObj)
        
        if isinstance(self.m_input,CGM2analysis.Analysis):
            print "AnaysisStats object"
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")
            
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_kineticFlag = kineticFlag             
        
        self.__translators=dict()
        self.__translators["Left.PelvisProgress.Angles"] = "LPelvisAngles"
        self.__translators["Left.Hip.Angles"] = "LHipAngles"
        self.__translators["Left.Knee.Angles"] = "LKneeAngles"            
        self.__translators["Left.Ankle.Angles"] = "LAnkleAngles"            
        self.__translators["Left.FootProgress.Angles"] = "LFootProgressAngles"               
        
        self.__translators["Right.PelvisProgress.Angles"] = "RPelvisAngles"
        self.__translators["Right.Hip.Angles"] = "RHipAngles"
        self.__translators["Right.Knee.Angles"] = "RKneeAngles"   
        self.__translators["Right.Ankle.Angles"] = "RAnkleAngles"            
        self.__translators["Right.FootProgress.Angles"] = "RFootProgressAngles"                  
   
        self.__translators["Left.Hip.Moment"] = "LHipMoment"
        self.__translators["Left.Knee.Moment"] = "LKneeMoment"
        self.__translators["Left.Ankle.Moment"] = "LAnkleMoment"
   
        self.__translators["Right.Hip.Moment"] = "RHipMoment"
        self.__translators["Right.Knee.Moment"] = "RKneeMoment"
        self.__translators["Right.Ankle.Moment"] = "RAnkleMoment"


        self.__translators["Left.Hip.Power"] = "LHipPower"
        self.__translators["Left.Knee.Power"] = "LKneePower"
        self.__translators["Left.Ankle.Power"] = "LAnklePower"
   
        self.__translators["Right.Hip.Power"] = "RHipPower"
        self.__translators["Right.Knee.Power"] = "RKneePower"
        self.__translators["Right.Ankle.Power"] = "RAnklePower"

        self.__limits=dict()
        for side in ["Left","Right"]:
            self.__limits[str(side)+".PelvisProgress.Angles","Tilt"] = None
            self.__limits[str(side)+".PelvisProgress.Angles","Obli"] = None
            self.__limits[str(side)+".PelvisProgress.Angles","Rota"] = None        
        
            self.__limits[str(side)+".Hip.Angles","Flex"] = None
            self.__limits[str(side)+".Hip.Angles","Addu"] = None
            self.__limits[str(side)+".Hip.Angles","Rota"] = None

            self.__limits[str(side)+".Knee.Angles","Flex"] = None
            self.__limits[str(side)+".Knee.Angles","Addu"] = None
            self.__limits[str(side)+".Knee.Angles","Rota"] = None

            self.__limits[str(side)+".Ankle.Angles","Flex"] = None
            self.__limits[str(side)+".Ankle.Angles","Addu"] = None
            
            self.__limits[str(side)+".FootProgress.Angles","Rota"] = None
                

        self.__staticLimit=dict()
        for side in ["Left","Right"]:
            self.__staticLimit[str(side)+".PelvisProgress.Angles","Tilt"] = None
            self.__staticLimit[str(side)+".PelvisProgress.Angles","Obli"] = None
            self.__staticLimit[str(side)+".PelvisProgress.Angles","Rota"] = None        
        
            self.__staticLimit[str(side)+".Hip.Angles","Flex"] = None
            self.__staticLimit[str(side)+".Hip.Angles","Addu"] = None
            self.__staticLimit[str(side)+".Hip.Angles","Rota"] = None

            self.__staticLimit[str(side)+".Knee.Angles","Flex"] = None
            self.__staticLimit[str(side)+".Knee.Angles","Addu"] = None
            self.__staticLimit[str(side)+".Knee.Angles","Rota"] = None

            self.__staticLimit[str(side)+".Ankle.Angles","Flex"] = None
            self.__staticLimit[str(side)+".Ankle.Angles","Addu"] = None
            
            self.__staticLimit[str(side)+".FootProgress.Angles","Rota"] = None       
       
    
    def printTranslators(self):
        for key in self.__translators.keys():
            print key
            

    def setTranslators(self,keys,newNames):
        if not isinstance(newNames,list) or not isinstance(keys,list):
            raise Exception ( "[pyCGM2] input parameter must be a list ")
        else:    
            if len(keys) != len(newNames):
                raise Exception( "[pyCGM2] input argument unbalanced")
            else:
                for key,name in zip(keys,newNames):
                    if key not in self.__translators.keys():
                        raise Exception( "key %s doesn t exit")
                    else:
                        self.__translators[key] = name
                    
    def setTranslator(self,key,newName):
        if key not in self.__translators.keys():
            raise Exception( "[pyCGM2] key %s doesn t exit")
        else:
            self.__translators[key] = newName
                    

    def setLimits(self,keyLabel,keyAxis,values):
        if not isinstance(values,list):
            raise Exception( "[pyCGM2] input values must be a list ")
        else:    
            for key in self.__limits.keys():
                flag = False                
                if key[0] == keyLabel and key[1]==keyAxis:
                    self.__limits[keyLabel,keyAxis] = values
                    flag = True
                    break
            if not flag:
                raise Exception( "[pyCGM2] check your input ( keyAngle or keyAxis not found) ")
                
                
    def setStaticLimit(self,keyLabel,keyAxis,value):
        if isinstance(value,int) or  isinstance(value,float) :
            for key in self.__staticLimit.keys():
                flag = False                
                if key[0] == keyLabel and key[1]==keyAxis:
                    self.__staticLimit[keyLabel,keyAxis] = value
                    flag = True
                    break
            if not flag:
                raise Exception( "[pyCGM2] check your input ( keyAngle or keyAxis not found) ")

        else:    
            raise Exception( "[pyCGM2] input value must be either a integer or a float ")
           

   
    def plotPanel(self,path,pdfSuffix, consistencyOnly=False,normativeData=None):
        
        pdfSuffixPlus = "_" + pdfSuffix if pdfSuffix != None else ""
        
        path = path if  path != None else ""

        if not consistencyOnly:
            # ---- descriptive Kinematics (mean + std)        
            figDescritiveKinematics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
            title=u""" Descriptive Kinematics \n """ 
            figDescritiveKinematics.suptitle(title)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
            
            ax1 = plt.subplot(5,3,1)# Pelvis X
            ax2 = plt.subplot(5,3,2)# Pelvis Y
            ax3 = plt.subplot(5,3,3)# Pelvis Z       
            ax4 = plt.subplot(5,3,4)# Hip X
            ax5 = plt.subplot(5,3,5)# Hip Y
            ax6 = plt.subplot(5,3,6)# Hip Z   
            ax7 = plt.subplot(5,3,7)# Knee X
            ax8 = plt.subplot(5,3,8)# Knee Y
            ax9 = plt.subplot(5,3,9)# Knee Z   
            ax10 = plt.subplot(5,3,10)# Ankle X
            ax11 = plt.subplot(5,3,11)# Ankle Y
            ax12 = plt.subplot(5,3,12)# Footprogress Z   
     

            axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]       
           
            # make
            suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else "" 
            gaitDescriptivePlot(ax1,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",0,"Pelvis Tilt", ylabel = " angle (deg)", 
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Tilt"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Tilt"],
                            leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Tilt"],rightStaticLimit = self.__staticLimit["Right.PelvisProgress.Angles","Tilt"])
            
            gaitDescriptivePlot(ax2,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",1,"Pelvis Obliquity", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Obli"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Obli"],
                            leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Obli"],rightStaticLimit=self.__staticLimit["Right.PelvisProgress.Angles","Obli"])

            gaitDescriptivePlot(ax3,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",2,"Pelvis Rotation", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Rota"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Rota"],
                            leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Rota"],rightStaticLimit=self.__staticLimit["Right.PelvisProgress.Angles","Rota"])

            gaitDescriptivePlot(ax4,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",0,"Hip flexion", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Flex"],rightLimits = self.__limits["Right.Hip.Angles","Flex"],
                            leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Hip.Angles","Flex"])

            gaitDescriptivePlot(ax5,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",1,"Hip adduction",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Addu"],rightLimits = self.__limits["Right.Hip.Angles","Addu"],
                            leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Addu"],rightStaticLimit=self.__staticLimit["Right.Hip.Angles","Addu"])

            gaitDescriptivePlot(ax6,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",2,"Hip rotation",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Rota"],rightLimits = self.__limits["Right.Hip.Angles","Rota"],
                            leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Rota"],rightStaticLimit= self.__staticLimit["Right.Hip.Angles","Rota"])

            gaitDescriptivePlot(ax7,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",0,"Knee flexion",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Flex"],rightLimits = self.__limits["Right.Knee.Angles","Flex"],
                            leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Knee.Angles","Flex"])

            gaitDescriptivePlot(ax8,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",1,"Knee adduction",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Addu"],rightLimits = self.__limits["Right.Knee.Angles","Addu"],
                            leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Addu"],rightStaticLimit = self.__staticLimit["Right.Knee.Angles","Addu"])

            gaitDescriptivePlot(ax9,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",2,"Knee rotation",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Rota"],rightLimits = self.__limits["Right.Knee.Angles","Rota"],
                            leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Rota"],rightStaticLimit = self.__staticLimit["Right.Knee.Angles","Rota"])


            gaitDescriptivePlot(ax10, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",0,"Ankle dorsiflexion",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.Ankle.Angles","Flex"],rightLimits = self.__limits["Right.Ankle.Angles","Flex"],
                            leftStaticLimit= self.__staticLimit["Left.Ankle.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Ankle.Angles","Flex"])

            gaitDescriptivePlot(ax11, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",1,"Ankle adduction",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.Ankle.Angles","Addu"],rightLimits = self.__limits["Right.Ankle.Angles","Addu"],
                            leftStaticLimit= self.__staticLimit["Left.Ankle.Angles","Addu"],rightStaticLimit= self.__staticLimit["Right.Ankle.Angles","Addu"])

            gaitDescriptivePlot(ax12, self.m_input.kinematicStats,  self.__translators["Left.FootProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.FootProgress.Angles"]+suffixPlus,"Right",1,"Foot Progression",ylabel = "angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.FootProgress.Angles","Rota"],rightLimits = self.__limits["Right.FootProgress.Angles","Rota"],
                            leftStaticLimit= self.__staticLimit["Left.FootProgress.Angles","Rota"],rightStaticLimit = self.__staticLimit["Right.FootProgress.Angles","Rota"])
 
            
   
            if normativeData is not None:
                ax1.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,0]-normativeData["Pelvis.Angles"]["sd"][:,0], normativeData["Pelvis.Angles"]["mean"][:,0]+normativeData["Pelvis.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax2.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,1]-normativeData["Pelvis.Angles"]["sd"][:,1], normativeData["Pelvis.Angles"]["mean"][:,1]+normativeData["Pelvis.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                ax3.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,2]-normativeData["Pelvis.Angles"]["sd"][:,2], normativeData["Pelvis.Angles"]["mean"][:,2]+normativeData["Pelvis.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
                ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,0]-normativeData["Hip.Angles"]["sd"][:,0], normativeData["Hip.Angles"]["mean"][:,0]+normativeData["Hip.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax5.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,1]-normativeData["Hip.Angles"]["sd"][:,1], normativeData["Hip.Angles"]["mean"][:,1]+normativeData["Hip.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                ax6.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,2]-normativeData["Hip.Angles"]["sd"][:,2], normativeData["Hip.Angles"]["mean"][:,2]+normativeData["Hip.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
                ax7.fill_between(np.linspace(0,100,51), normativeData["Knee.Angles"]["mean"][:,0]-normativeData["Knee.Angles"]["sd"][:,0], normativeData["Knee.Angles"]["mean"][:,0]+normativeData["Knee.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,0]-normativeData["Ankle.Angles"]["sd"][:,0], normativeData["Ankle.Angles"]["mean"][:,0]+normativeData["Ankle.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,2]-normativeData["Ankle.Angles"]["sd"][:,2], normativeData["Ankle.Angles"]["mean"][:,2]+normativeData["Ankle.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

    
            pp = PdfPages(str(path+ "descriptiveKinematics"+pdfSuffixPlus+".pdf"))
            pp.savefig(figDescritiveKinematics)    
            pp.close() 
        
        
        # ---- consitency Kinematics plot      
        figConsistencyKinematics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Consistency Kinematics \n """  
        figConsistencyKinematics.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        
        ax1 = plt.subplot(5,3,1)# Pelvis X
        ax2 = plt.subplot(5,3,2)# Pelvis Y
        ax3 = plt.subplot(5,3,3)# Pelvis Z       
        ax4 = plt.subplot(5,3,4)# Hip X
        ax5 = plt.subplot(5,3,5)# Hip Y
        ax6 = plt.subplot(5,3,6)# Hip Z   
        ax7 = plt.subplot(5,3,7)# Knee X
        ax8 = plt.subplot(5,3,8)# Knee Y
        ax9 = plt.subplot(5,3,9)# Knee Z   
        ax10 = plt.subplot(5,3,10)# Ankle X
        ax11 = plt.subplot(5,3,11)# Ankle Y
        ax12 = plt.subplot(5,3,12)# FootProgress   


        axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]       
       
        # make
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""  
        gaitConsistencyPlot(ax1,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",0,"Pelvis Tilt", ylabel = " angle (deg)", 
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Tilt"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Tilt"],
                            leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Tilt"],rightStaticLimit = self.__staticLimit["Right.PelvisProgress.Angles","Tilt"])
            
        gaitConsistencyPlot(ax2,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",1,"Pelvis Obliquity", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.PelvisProgress.Angles","Obli"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Obli"],
                        leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Obli"],rightStaticLimit=self.__staticLimit["Right.PelvisProgress.Angles","Obli"])

        gaitConsistencyPlot(ax3,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",2,"Pelvis Rotation", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.PelvisProgress.Angles","Rota"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Rota"],
                        leftStaticLimit= self.__staticLimit["Left.PelvisProgress.Angles","Rota"],rightStaticLimit=self.__staticLimit["Right.PelvisProgress.Angles","Rota"])

        gaitConsistencyPlot(ax4,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",0,"Hip flexion", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Flex"],rightLimits = self.__limits["Right.Hip.Angles","Flex"],
                        leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Hip.Angles","Flex"])

        gaitConsistencyPlot(ax5,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",1,"Hip adduction",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Addu"],rightLimits = self.__limits["Right.Hip.Angles","Addu"],
                        leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Addu"],rightStaticLimit=self.__staticLimit["Right.Hip.Angles","Addu"])

        gaitConsistencyPlot(ax6,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",2,"Hip rotation",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Rota"],rightLimits = self.__limits["Right.Hip.Angles","Rota"],
                        leftStaticLimit= self.__staticLimit["Left.Hip.Angles","Rota"],rightStaticLimit= self.__staticLimit["Right.Hip.Angles","Rota"])

        gaitConsistencyPlot(ax7,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",0,"Knee flexion",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Flex"],rightLimits = self.__limits["Right.Knee.Angles","Flex"],
                        leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Knee.Angles","Flex"])

        gaitConsistencyPlot(ax8,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",1,"Knee adduction",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Addu"],rightLimits = self.__limits["Right.Knee.Angles","Addu"],
                        leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Addu"],rightStaticLimit = self.__staticLimit["Right.Knee.Angles","Addu"])

        gaitConsistencyPlot(ax9,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",2,"Knee rotation",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Rota"],rightLimits = self.__limits["Right.Knee.Angles","Rota"],
                        leftStaticLimit= self.__staticLimit["Left.Knee.Angles","Rota"],rightStaticLimit = self.__staticLimit["Right.Knee.Angles","Rota"])

        gaitConsistencyPlot(ax10, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",0,"Ankle dorsiflexion",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.Ankle.Angles","Flex"],rightLimits = self.__limits["Right.Ankle.Angles","Flex"],
                        leftStaticLimit= self.__staticLimit["Left.Ankle.Angles","Flex"],rightStaticLimit=self.__staticLimit["Right.Ankle.Angles","Flex"])

        gaitConsistencyPlot(ax11, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",1,"Ankle adduction",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.Ankle.Angles","Addu"],rightLimits = self.__limits["Right.Ankle.Angles","Addu"],
                        leftStaticLimit= self.__staticLimit["Left.Ankle.Angles","Addu"],rightStaticLimit= self.__staticLimit["Right.Ankle.Angles","Addu"])

        gaitConsistencyPlot(ax12, self.m_input.kinematicStats,  self.__translators["Left.FootProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.FootProgress.Angles"]+suffixPlus,"Right",1,"Foot Progression",ylabel = "angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.FootProgress.Angles","Rota"],rightLimits = self.__limits["Right.FootProgress.Angles","Rota"],
                        leftStaticLimit= self.__staticLimit["Left.FootProgress.Angles","Rota"],rightStaticLimit = self.__staticLimit["Right.FootProgress.Angles","Rota"])        
        
        
        if normativeData is not None:
            ax1.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,0]-normativeData["Pelvis.Angles"]["sd"][:,0], normativeData["Pelvis.Angles"]["mean"][:,0]+normativeData["Pelvis.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax2.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,1]-normativeData["Pelvis.Angles"]["sd"][:,1], normativeData["Pelvis.Angles"]["mean"][:,1]+normativeData["Pelvis.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
            ax3.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,2]-normativeData["Pelvis.Angles"]["sd"][:,2], normativeData["Pelvis.Angles"]["mean"][:,2]+normativeData["Pelvis.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
            ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,0]-normativeData["Hip.Angles"]["sd"][:,0], normativeData["Hip.Angles"]["mean"][:,0]+normativeData["Hip.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax5.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,1]-normativeData["Hip.Angles"]["sd"][:,1], normativeData["Hip.Angles"]["mean"][:,1]+normativeData["Hip.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
            ax6.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,2]-normativeData["Hip.Angles"]["sd"][:,2], normativeData["Hip.Angles"]["mean"][:,2]+normativeData["Hip.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
            ax7.fill_between(np.linspace(0,100,51), normativeData["Knee.Angles"]["mean"][:,0]-normativeData["Knee.Angles"]["sd"][:,0], normativeData["Knee.Angles"]["mean"][:,0]+normativeData["Knee.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

            ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,0]-normativeData["Ankle.Angles"]["sd"][:,0], normativeData["Ankle.Angles"]["mean"][:,0]+normativeData["Ankle.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,2]-normativeData["Ankle.Angles"]["sd"][:,2], normativeData["Ankle.Angles"]["mean"][:,2]+normativeData["Ankle.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)


        pp = PdfPages(str(path+ "consistencyKinematics"+pdfSuffixPlus+".pdf"))
        pp.savefig(figConsistencyKinematics)    
        pp.close() 
        
        if self.m_kineticFlag:
            if not consistencyOnly:
                # ---- descriptive Kinetics (mean + std)        
                figDescriptiveKinetics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
                title=u""" Descriptive Kinetics \n """ 
                figDescriptiveKinetics.suptitle(title)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                
                ax1 = plt.subplot(3,4,1)# Hip X extensor
                ax2 = plt.subplot(3,4,2)# Hip Y abductor
                ax3 = plt.subplot(3,4,3)# Hip Z rotation      
                ax4 = plt.subplot(3,4,4)# Knee Z power
        
                ax5 = plt.subplot(3,4,5)# knee X extensor
                ax6 = plt.subplot(3,4,6)# knee Y abductor
                ax7 = plt.subplot(3,4,7)# knee Z rotation      
                ax8 = plt.subplot(3,4,8)# knee Z power
        
                ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
                ax10 = plt.subplot(3,4,10)# ankle Y rotation
                ax11 = plt.subplot(3,4,11)# ankle Z everter      
                ax12 = plt.subplot(3,4,12)# ankle Z power
        
        
                axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]       
               
                #  make
                suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""  
               
                gaitDescriptivePlot(ax1,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",0,"Hip extensor Moment", ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax2,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",1,"Hip abductor Moment",ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax3,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",2,"Hip rotation Moment",ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax4,self.m_input.kineticStats, self.__translators["Left.Hip.Power"]+suffixPlus, "Left", self.__translators["Right.Hip.Power"]+suffixPlus,"Right",2,"Hip power",ylabel = " power (W.kg-1)")
         
                gaitDescriptivePlot(ax5,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",0,"Knee extensor Moment", ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax6,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",1,"Knee abductor Moment",ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax7,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",2,"Knee rotation Moment",ylabel = " moment (N.mm.kg-1)")
                gaitDescriptivePlot(ax8,self.m_input.kineticStats, self.__translators["Left.Knee.Power"]+suffixPlus, "Left",self.__translators["Right.Knee.Power"]+suffixPlus,"Right",2,"Knee power",ylabel = " power (W.kg-1)")
        
                gaitDescriptivePlot(ax9,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",0,"Ankle Plantarflexor Moment", ylabel = " moment (N.mm.kg-1)", xlabel ="Gait cycle %")
                gaitDescriptivePlot(ax10,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",1,"Ankle rotation Moment",ylabel = " moment (N.mm.kg-1)",  xlabel ="Gait cycle %")
                gaitDescriptivePlot(ax11,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",2,"Ankle everter Moment",ylabel = " moment (N.mm.kg-1)", xlabel ="Gait cycle %")
                gaitDescriptivePlot(ax12,self.m_input.kineticStats, self.__translators["Left.Ankle.Power"]+suffixPlus,"Left", self.__translators["Right.Ankle.Power"]+suffixPlus,"Right",2,"Ankle power",ylabel = " power (W.kg-1)", xlabel ="Gait cycle %")
        
                if normativeData is not None:
                    ax1.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,0]-normativeData["Hip.Moment"]["sd"][:,0], normativeData["Hip.Moment"]["mean"][:,0]+normativeData["Hip.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax2.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,1]-normativeData["Hip.Moment"]["sd"][:,1], normativeData["Hip.Moment"]["mean"][:,1]+normativeData["Hip.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                    
                    ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Power"]["mean"][:,2]-normativeData["Hip.Power"]["sd"][:,2], normativeData["Hip.Power"]["mean"][:,2]+normativeData["Hip.Power"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                    ax5.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,0]-normativeData["Knee.Moment"]["sd"][:,0], normativeData["Knee.Moment"]["mean"][:,0]+normativeData["Knee.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax6.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,1]-normativeData["Knee.Moment"]["sd"][:,1], normativeData["Knee.Moment"]["mean"][:,1]+normativeData["Knee.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                    ax8.fill_between(np.linspace(0,100,51), normativeData["Knee.Power"]["mean"][:,2]-normativeData["Knee.Power"]["sd"][:,2], normativeData["Knee.Power"]["mean"][:,2]+normativeData["Knee.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

                    ax9.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,0]-normativeData["Ankle.Moment"]["sd"][:,0], normativeData["Ankle.Moment"]["mean"][:,0]+normativeData["Ankle.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,2]-normativeData["Ankle.Moment"]["sd"][:,1], normativeData["Ankle.Moment"]["mean"][:,1]+normativeData["Ankle.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                    ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Power"]["mean"][:,2]-normativeData["Ankle.Power"]["sd"][:,2], normativeData["Ankle.Power"]["mean"][:,2]+normativeData["Ankle.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

        
                pp = PdfPages(str(path+ "descriptiveKinetics"+pdfSuffixPlus+".pdf"))
                pp.savefig(figDescriptiveKinetics)    
                pp.close() 
            
            
            # ---- consistency Kinetics (mean + std)        
            figConsistencyKinetics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
            title=u""" Consistency Kinetics \n """ 
            figConsistencyKinetics.suptitle(title)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
            
            ax1 = plt.subplot(3,4,1)# Hip X extensor
            ax2 = plt.subplot(3,4,2)# Hip Y abductor
            ax3 = plt.subplot(3,4,3)# Hip Z rotation      
            ax4 = plt.subplot(3,4,4)# Knee Z power
    
            ax5 = plt.subplot(3,4,5)# knee X extensor
            ax6 = plt.subplot(3,4,6)# knee Y abductor
            ax7 = plt.subplot(3,4,7)# knee Z rotation      
            ax8 = plt.subplot(3,4,8)# knee Z power
    
            ax9 = plt.subplot(3,4,9)# amkle X plantar flexion
            ax10 = plt.subplot(3,4,10)# ankle Y rotation
            ax11 = plt.subplot(3,4,11)# ankle Z everter      
            ax12 = plt.subplot(3,4,12)# ankle Z power
    
    
            axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]       
           
            #  make
            suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else "" 
            gaitConsistencyPlot(ax1,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",0,"Hip extensor Moment", ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax2,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",1,"Hip abductor Moment",ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax3,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",2,"Hip rotation Moment",ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax4,self.m_input.kineticStats, self.__translators["Left.Hip.Power"]+suffixPlus, "Left", self.__translators["Right.Hip.Power"]+suffixPlus,"Right",2,"Hip power",ylabel = " power (W.kg-1)")
     
            gaitConsistencyPlot(ax5,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",0,"Knee extensor Moment", ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax6,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",1,"Knee abductor Moment",ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax7,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",2,"Knee rotation Moment",ylabel = " moment (N.mm.kg-1)")
            gaitConsistencyPlot(ax8,self.m_input.kineticStats, self.__translators["Left.Knee.Power"]+suffixPlus,"Left", self.__translators["Right.Knee.Power"]+suffixPlus,"Right",2,"Knee power",ylabel = " power (W.kg-1)")
    
            gaitConsistencyPlot(ax9,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",0,"Ankle Plantarflexor Moment", ylabel = " moment (N.mm.kg-1)", xlabel ="Gait cycle %")
            gaitConsistencyPlot(ax10,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",1,"Ankle rotation Moment",ylabel = " moment (N.mm.kg-1)", xlabel ="Gait cycle %")
            gaitConsistencyPlot(ax11,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",2,"Ankle everter Moment",ylabel = " moment (N.mm.kg-1)", xlabel ="Gait cycle %")
            gaitConsistencyPlot(ax12,self.m_input.kineticStats, self.__translators["Left.Ankle.Power"]+suffixPlus, "Left", self.__translators["Right.Ankle.Power"]+suffixPlus,"Right",2,"Ankle power",ylabel = " power (W.kg-1)", xlabel ="Gait cycle %")
    
    
            if normativeData is not None:
                ax1.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,0]-normativeData["Hip.Moment"]["sd"][:,0], normativeData["Hip.Moment"]["mean"][:,0]+normativeData["Hip.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax2.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,1]-normativeData["Hip.Moment"]["sd"][:,1], normativeData["Hip.Moment"]["mean"][:,1]+normativeData["Hip.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                
                ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Power"]["mean"][:,2]-normativeData["Hip.Power"]["sd"][:,2], normativeData["Hip.Power"]["mean"][:,2]+normativeData["Hip.Power"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                ax5.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,0]-normativeData["Knee.Moment"]["sd"][:,0], normativeData["Knee.Moment"]["mean"][:,0]+normativeData["Knee.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax6.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,1]-normativeData["Knee.Moment"]["sd"][:,1], normativeData["Knee.Moment"]["mean"][:,1]+normativeData["Knee.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                ax8.fill_between(np.linspace(0,100,51), normativeData["Knee.Power"]["mean"][:,2]-normativeData["Knee.Power"]["sd"][:,2], normativeData["Knee.Power"]["mean"][:,2]+normativeData["Knee.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

                ax9.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,0]-normativeData["Ankle.Moment"]["sd"][:,0], normativeData["Ankle.Moment"]["mean"][:,0]+normativeData["Ankle.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,2]-normativeData["Ankle.Moment"]["sd"][:,1], normativeData["Ankle.Moment"]["mean"][:,1]+normativeData["Ankle.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Power"]["mean"][:,2]-normativeData["Ankle.Power"]["sd"][:,2], normativeData["Ankle.Power"]["mean"][:,2]+normativeData["Ankle.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)    
    
            pp = PdfPages(str(path+ "consistencyKinetics"+pdfSuffixPlus+".pdf"))
    
            pp.savefig(figConsistencyKinetics)    
            pp.close() 




  

