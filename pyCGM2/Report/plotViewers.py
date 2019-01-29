# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# pyCGM2
import pyCGM2
from pyCGM2.Report import plot
from pyCGM2 import ma



class AbstractPlotViewer(object):
    """
    Abstract Builder
    """
    def __init__(self,input):
        self.m_input =input

    def setNormativeData(self):
        pass

    def __setLayer(self):
        pass

    def __setData(self):
        pass

    def plotPanel(self):
        pass

class SpatioTemporalPlotViewer(AbstractPlotViewer):
    """


    """

    def __init__(self,iAnalysis):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
        """

        super(SpatioTemporalPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setLayer(self):
        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Spatio temporal Gait Parameters \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(8,1,1)# speed overall (speed)
        ax1 = plt.subplot(8,1,2)# length overall (strideLength)
        ax2 = plt.subplot(8,1,3)# width overall (strideWidth
        ax3 = plt.subplot(8,1,4)# cadence overall (cadence)

        ax4 = plt.subplot(8,1,5)# Step length  steplength
        ax5 = plt.subplot(8,1,6)# SStep time (s) duration
        ax6 = plt.subplot(8,1,7)# Stance time (% of gait cycle) stancePhase
        ax7 = plt.subplot(8,1,8)# Initial double limb support (s) doubleStance1


    def __setData(self):

        plot.stpHorizontalHistogram(self.fig.axes[0],self.m_analysis.stpStats,
                                "speed",
                                overall= True,
                                title="speed (m.s-1)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[1],self.m_analysis.stpStats,
                                "strideLength",
                                overall= True,
                                title="stride length (m)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[2],self.m_analysis.stpStats,
                                "strideWidth",
                                overall= True,
                                title="stride width (m)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[3],self.m_analysis.stpStats,
                                "cadence",
                                overall= True,
                                title="cadence", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[4],self.m_analysis.stpStats,
                                "stepLength",
                                overall= False,
                                title="Step length", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[5],self.m_analysis.stpStats,
                                "duration",
                                overall= False,
                                title="Step time", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[6],self.m_analysis.stpStats,
                                "stancePhase",
                                overall= False,
                                title="Stance time (% of gait cycle)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[7],self.m_analysis.stpStats,
                                "doubleStance1",
                                overall= False,
                                title="Initial double limb support (% of gait cycle)", xlabel=None,xlim=None)


    def plotPanel(self):
        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            labels = ["speed", "stride length", "stride width", "cadence", "step length ","step time", "stance phase","initial double stance"]

            i=0
            for label in labels:
                minusStd = self.m_normativeData[label]["Mean"]-self.m_normativeData[label]["Std"]
                plusStd = self.m_normativeData[label]["Mean"]+self.m_normativeData[label]["Std"]
                self.fig.axes[i].axvline(x=(minusStd), color= "green", linestyle = "dashed")
                self.fig.axes[i].axvline(x=(plusStd), color= "green", linestyle = "dashed")
                i+=1



class GpsMapPlotViewer(AbstractPlotViewer):
    """


    """

    def __init__(self,iAnalysis):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
        """

        super(GpsMapPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        pass

    def __setLayer(self):
        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u"""Mouvement  Analysis Profile \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(1,1,1)


    def __setData(self):

        N = 9
        overall = (self.m_analysis.gvs["LPelvisAngles","Left"]["mean"][0],
                     0,
                     0,
                     self.m_analysis.gvs["LPelvisAngles","Left"]["mean"][1],
                     0,
                     self.m_analysis.gvs["LPelvisAngles","Left"]["mean"][2],
                     0,
                     0,
                      self.m_analysis.gps["Overall"]["mean"][0])
        left = (0,
                 self.m_analysis.gvs["LHipAngles","Left"]["mean"][0],
                 self.m_analysis.gvs["LKneeAngles","Left"]["mean"][0],
                  0,
                  self.m_analysis.gvs["LHipAngles","Left"]["mean"][1],
                  0,
                  self.m_analysis.gvs["LHipAngles","Left"]["mean"][2],
                  self.m_analysis.gvs["LFootProgressAngles","Left"]["mean"][2],
                  self.m_analysis.gps["Context"]["Left"]["mean"][0])

        right = (0,
                 self.m_analysis.gvs["RHipAngles","Right"]["mean"][0],
                 self.m_analysis.gvs["RKneeAngles","Right"]["mean"][0],
                  0,
                  self.m_analysis.gvs["RHipAngles","Right"]["mean"][1],
                  0,
                  self.m_analysis.gvs["RHipAngles","Right"]["mean"][2],
                  self.m_analysis.gvs["RFootProgressAngles","Right"]["mean"][2],
                  self.m_analysis.gps["Context"]["Right"]["mean"][0])

        width = 0.35
        y_pos = [0,2,4,6,8,10,12,14,16]
        y_pos1 = [i+0.35 for i in y_pos]
        y_pos2 =  [i+0.7 for i in y_pos]
        self.fig.axes[0].bar(y_pos, overall, width, label='Overall',color='purple')
        self.fig.axes[0].bar(y_pos1 , left, width,  label='Left', color='red')
        self.fig.axes[0].bar(y_pos2 , right, width,  label='Right',color='blue')

        self.fig.axes[0].set_ylabel('Scores (deg)')

        tickLabels = [i+width / 3.0 for i in y_pos]
        self.fig.axes[0].set_xticks(tickLabels)
        self.fig.axes[0].set_xticklabels(('Pelvis Ant/ret', 'Hip Flex',  'Knee Flex',
                                       'Pelvis Up/Down', 'Hip Abd',
                                       'Pelvis Rot', 'Hip Rot',"Foot Prog",
                                       "GPS"), rotation=45)
        self.fig.axes[0].legend(loc='best')



    def plotPanel(self):
        self.__setLayer()
        self.__setData()


class LowerLimbKinematicsPlotViewer(AbstractPlotViewer):
    """
        **Description :** Constructor of gait plot panel.

        .. note::

            The kinematic panel is made of 12 subplots

            ================  ==============================  =======  ===========
            matplotlib Axis   model outputs                   context  Axis label
            ================  ==============================  =======  ===========
            ax1               "LPelvisProgressAngles"         left     Tilt
                              "RPelvisProgressAngles"         right
            ax2               "LPelvisProgress.Angles"        left     Obli
                              "RPelvisProgress.Angles"        right
            ax3               "LPelvisProgress.Angles"        left     Rota
                              "RPelvisProgress.Angles"        right
            ax4               "LHipAngles"                    left     Flex
                              "RHipAngles"                    right
            ax5               "LHipAngles"                    left     Addu
                              "RHipAngles"                    right
            ax6               "LHipAngles"                    left    Rota
                              "RHipAngles"                    right
            ax7               "LKneeAngles"                   left     Flex
                              "RKneeAngles"                   right
            ax8               "LKneeAngles"                   left     Addu
                              "RKneeAngles"                   right
            ax9               "LKneeAngles"                   left     Rota
                              "RKneeAngles"                   right
            ax10              "LAnkleAngles"                  left     Flex
                              "RAnkleAngles"                  right
            ax11              "LAnkleAngles"                  left     Eve
                              "RAnkleAngles"                  right
            ax12              "LFootProgressAngles"           left
                              "RFootProgressAngles"           right
            ax13              "LForeFoot"                     left      Flex
                               "RForeFoot"                   right
            ax14              "LForeFoot"                     left      Rota
                               "RForeFoot"                   right
            ax15              "LForeFoot"                     left      Addu
                               "RForeFoot"                   right
            ================  ==============================  =======  ===========

    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(LowerLimbKinematicsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")


        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_concretePlotFunction = None

    def setConcretePlotFunction(self, concreteplotFunction):
        self.m_concretePlotFunction = concreteplotFunction


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")

        if self.m_concretePlotFunction.func_name in ["descriptivePlot","gaitDescriptivePlot"]:
            title=u""" Descriptive Time-normalized Kinematics \n """
        elif self.m_concretePlotFunction.func_name in ["consistencyPlot","gaitConsistencyPlot"]:
            title=u""" Consistency Time-normalized Kinematics \n """
        else :
            title=u"""\n"""

        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(6,3,1)# Pelvis X
        ax1 = plt.subplot(6,3,2)# Pelvis Y
        ax2 = plt.subplot(6,3,3)# Pelvis Z
        ax3 = plt.subplot(6,3,4)# Hip X
        ax4 = plt.subplot(6,3,5)# Hip Y
        ax5 = plt.subplot(6,3,6)# Hip Z
        ax6 = plt.subplot(6,3,7)# Knee X
        ax7 = plt.subplot(6,3,8)# Knee Y
        ax8 = plt.subplot(6,3,9)# Knee Z
        ax9 = plt.subplot(6,3,10)# Ankle X
        ax10 = plt.subplot(6,3,11)# Ankle Z
        ax11 = plt.subplot(6,3,12)# Footprogress Z
        ax12 = plt.subplot(6,3,13)# ForeFoot X
        ax13 = plt.subplot(6,3,14)# ForeFoot Z
        ax14 = plt.subplot(6,3,15)# ForeFoot Y

        ax0.set_title("Pelvis Tilt" ,size=8)
        ax1.set_title("Pelvis Obliquity" ,size=8)
        ax2.set_title("Pelvis Rotation" ,size=8)
        ax3.set_title("Hip Flexion" ,size=8)
        ax4.set_title("Hip Adduction" ,size=8)
        ax5.set_title("Hip Rotation" ,size=8)
        ax6.set_title("Knee Flexion" ,size=8)
        ax7.set_title("Knee Adduction" ,size=8)
        ax8.set_title("Knee Rotation" ,size=8)
        ax9.set_title("Ankle dorsiflexion" ,size=8)
        ax10.set_title("Ankle supination" ,size=8)
        ax11.set_title("Foot Progression " ,size=8)
        ax12.set_title("ForeFoot dorsiflexion " ,size=8)
        ax13.set_title("ForeFoot eversion " ,size=8)
        ax14.set_title("ForeFoot Adduction " ,size=8)

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)

        ax12.set_xlabel("Cycle %",size=8)
        ax13.set_xlabel("Cycle %",size=8)
        ax14.set_xlabel("Cycle %",size=8)

        ax0.set_ylim([0,60])
        ax1.set_ylim([-30,30])
        ax2.set_ylim([-30,30])

        ax3.set_ylim( [-20,70])
        ax4.set_ylim([-30,30])
        ax5.set_ylim([-30,30])

        ax6.set_ylim([-15,75])
        ax7.set_ylim([-30,30])
        ax8.set_ylim([-30,30])

        ax9.set_ylim([-30,30])
        ax10.set_ylim([-30,30])
        ax11.set_ylim([-30,30])

        ax12.set_ylim([-50,30])
        ax13.set_ylim([-30,30])
        ax14.set_ylim([-30,30])

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        # pelvis
        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kinematicStats,
                "LPelvisAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kinematicStats,
                "RPelvisAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kinematicStats,
                "LPelvisAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kinematicStats,
                "RPelvisAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kinematicStats,
                "LPelvisAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kinematicStats,
                "RPelvisAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

        # hip
        self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                "LHipAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                "RHipAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
                "LHipAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
                "RHipAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
                "LHipAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
                "RHipAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

        # knee
        self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                "LKneeAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                "RKneeAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                "LKneeAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                "RKneeAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                "LKneeAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                "RKneeAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

        # ankle
        self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kinematicStats,
                "LAnkleAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kinematicStats,
                "RAnkleAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kinematicStats,
                "LAnkleAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kinematicStats,
                "RAnkleAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

        # foot progress
        self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kinematicStats,
                "LFootProgressAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kinematicStats,
                "RFootProgressAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

        # ForeFoot
        self.m_concretePlotFunction(self.fig.axes[12],self.m_analysis.kinematicStats,
                "LForeFootAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[12],self.m_analysis.kinematicStats,
                "RForeFootAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[13],self.m_analysis.kinematicStats,
                "LForeFootAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[13],self.m_analysis.kinematicStats,
                "RForeFootAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[14],self.m_analysis.kinematicStats,
                "LForeFootAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[14],self.m_analysis.kinematicStats,
                "RForeFootAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)


    def plotPanel(self):

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            self.fig.axes[0].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Pelvis.Angles"]["mean"][:,0]-self.m_normativeData["Pelvis.Angles"]["sd"][:,0],
                self.m_normativeData["Pelvis.Angles"]["mean"][:,0]+self.m_normativeData["Pelvis.Angles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[1].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Pelvis.Angles"]["mean"][:,1]-self.m_normativeData["Pelvis.Angles"]["sd"][:,1],
                self.m_normativeData["Pelvis.Angles"]["mean"][:,1]+self.m_normativeData["Pelvis.Angles"]["sd"][:,1],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[2].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Pelvis.Angles"]["mean"][:,2]-self.m_normativeData["Pelvis.Angles"]["sd"][:,2],
                self.m_normativeData["Pelvis.Angles"]["mean"][:,2]+self.m_normativeData["Pelvis.Angles"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)


            self.fig.axes[3].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Angles"]["mean"][:,0]-self.m_normativeData["Hip.Angles"]["sd"][:,0],
                self.m_normativeData["Hip.Angles"]["mean"][:,0]+self.m_normativeData["Hip.Angles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[4].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Angles"]["mean"][:,1]-self.m_normativeData["Hip.Angles"]["sd"][:,1],
                self.m_normativeData["Hip.Angles"]["mean"][:,1]+self.m_normativeData["Hip.Angles"]["sd"][:,1],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[5].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Angles"]["mean"][:,2]-self.m_normativeData["Hip.Angles"]["sd"][:,2],
                self.m_normativeData["Hip.Angles"]["mean"][:,2]+self.m_normativeData["Hip.Angles"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[6].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Knee.Angles"]["mean"][:,0]-self.m_normativeData["Knee.Angles"]["sd"][:,0],
                self.m_normativeData["Knee.Angles"]["mean"][:,0]+self.m_normativeData["Knee.Angles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[9].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Ankle.Angles"]["mean"][:,0]-self.m_normativeData["Ankle.Angles"]["sd"][:,0],
                self.m_normativeData["Ankle.Angles"]["mean"][:,0]+self.m_normativeData["Ankle.Angles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[11].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Foot.Angles"]["mean"][:,2]-self.m_normativeData["Ankle.Angles"]["sd"][:,2],
                self.m_normativeData["Foot.Angles"]["mean"][:,2]+self.m_normativeData["Ankle.Angles"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)

        return self.fig





class LowerLimbKineticsPlotViewer(AbstractPlotViewer):
    """
        **Description :** Constructor of gait plot panel.

        .. note::

            The kinematic panel is made of 12 subplots

            ================  ==============================  =======  ===========
            matplotlib Axis   model outputs                   context  Axis label
            ================  ==============================  =======  ===========
            ax1               "LHipMoment"                    left     Ext
                              "RHipMoment"                    right
            ax2               "LHipMoment"                    left     Abd
                              "RHipMoment"                    right
            ax3               "LHipMoment"                    left     Rot
                              "RHipMoment"                    right
            ax4               "LHipPower"                     left
                              "RHipPower"                     right
            ax5               "LKneeMoment"                   left     Ext
                              "RKneeMoment"                   right
            ax6               "LKneeMoment"                   left     Abd
                              "RKneeMoment"                   right
            ax7               "LKneeMoment"                   left     Rot
                              "RKneeMoment"                   right
            ax8               "LKneePower"                    left
                              "RKneePower"                    right
            ax9               "LAnkleMoment"                  left     Pla
                              "RAnkleMoment"                  right
            ax10              "LAnkleMoment"                  left     Eve
                              "RAnkleMoment"                  right
            ax11              "LAnkleMoment"                  left     Rot
                              "RAnkleMoment"                  right
            ax12              "LAnklePower"                   left
                              "RAnklePower"                   right
            ================  ==============================  =======  ===========


    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(LowerLimbKineticsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")


        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_concretePlotFunction = None

    def setConcretePlotFunction(self, concreteplotFunction):
        self.m_concretePlotFunction = concreteplotFunction

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        if self.m_concretePlotFunction.func_name in ["descriptivePlot","gaitDescriptivePlot"]:
            title=u""" Descriptive Time-normalized Kinetics \n """
        elif self.m_concretePlotFunction.func_name in ["consistencyPlot","gaitConsistencyPlot"]:
            title=u""" Consistency Time-normalized Kinetics \n """
        else :
            title=u"""\n"""
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(3,4,1)# Hip X extensor
        ax1 = plt.subplot(3,4,2)# Hip Y abductor
        ax2 = plt.subplot(3,4,3)# Hip Z rotation
        ax3 = plt.subplot(3,4,4)# Knee Z power

        ax4 = plt.subplot(3,4,5)# knee X extensor
        ax5 = plt.subplot(3,4,6)# knee Y abductor
        ax6 = plt.subplot(3,4,7)# knee Z rotation
        ax7 = plt.subplot(3,4,8)# knee Z power

        ax8 = plt.subplot(3,4,9)# ankle X plantar flexion
        ax9 = plt.subplot(3,4,10)# ankle Y rotation
        ax10 = plt.subplot(3,4,11)# ankle Z everter
        ax11 = plt.subplot(3,4,12)# ankle Z power

        ax0.set_title("Hip extensor Moment" ,size=8)
        ax1.set_title("Hip abductor Moment" ,size=8)
        ax2.set_title("Hip rotation Moment" ,size=8)
        ax3.set_title("Hip Power" ,size=8)

        ax4.set_title("Knee extensor Moment" ,size=8)
        ax5.set_title("Knee abductor Moment" ,size=8)
        ax6.set_title("Knee rotation Moment" ,size=8)
        ax7.set_title("Knee Power" ,size=8)

        ax8.set_title("Ankle plantarflexor Moment" ,size=8)
        ax9.set_title("Ankle everter Moment" ,size=8)
        ax10.set_title("Ankle abductor Moment" ,size=8)
        ax11.set_title("Ankle Power " ,size=8)

        for ax in [self.fig.axes[0],self.fig.axes[1],self.fig.axes[2],
                   self.fig.axes[4],self.fig.axes[5],self.fig.axes[6],
                   self.fig.axes[8],self.fig.axes[9],self.fig.axes[10]]:
            ax.set_ylabel("moment (N.mm.kg-1)",size=8)

        for ax in [self.fig.axes[3],self.fig.axes[7],self.fig.axes[8]]:
            ax.set_ylabel("power (W.Kg-1)",size=8)

        ax8.set_xlabel("Cycle %",size=8)
        ax9.set_xlabel("Cycle %",size=8)
        ax10.set_xlabel("Cycle %",size=8)
        ax11.set_xlabel("Cycle %",size=8)

        ax0.set_ylim([-2.0 *1000.0, 3.0*1000.0])
        ax1.set_ylim([-2.0*1000.0, 1.0*1000.0])
        ax2.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax3.set_ylim(  [-3.0, 3.0])

        ax4.set_ylim([-1.0*1000.0, 1.0*1000.0])
        ax5.set_ylim([-1.0*1000.0, 1.0*1000.0])
        ax6.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax7.set_ylim( [-3.0, 3.0])

        ax8.set_ylim([-1.0*1000.0, 3.0*1000.0])
        ax9.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax10.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax11.set_ylim( [-2.0, 5.0])


    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        # hip
        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kineticStats,
                "LHipMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kineticStats,
                "RHipMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kineticStats,
                "LHipMoment"+suffixPlus,"Left",1, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kineticStats,
                "RHipMoment"+suffixPlus,"Right",1, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kineticStats,
                "LHipMoment"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kineticStats,
                "RHipMoment"+suffixPlus,"Right",2, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kineticStats,
                "LHipPower"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kineticStats,
                "RHipPower"+suffixPlus,"Right",2, color="blue", customLimits=None)

        # knee
        self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kineticStats,
                "LKneeMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kineticStats,
                "RKneeMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kineticStats,
                "LKneeMoment"+suffixPlus,"Left",1, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kineticStats,
                "RKneeMoment"+suffixPlus,"Right",1, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kineticStats,
                "LKneeMoment"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kineticStats,
                "RKneeMoment"+suffixPlus,"Right",2, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kineticStats,
                "LKneePower"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kineticStats,
                "RKneePower"+suffixPlus,"Right",2, color="blue", customLimits=None)

        # ankle
        self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kineticStats,
                "LAnkleMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kineticStats,
                "RAnkleMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kineticStats,
                "LAnkleMoment"+suffixPlus,"Left",1, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kineticStats,
                "RAnkleMoment"+suffixPlus,"Right",1, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kineticStats,
                "LAnkleMoment"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kineticStats,
                "RAnkleMoment"+suffixPlus,"Right",2, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kineticStats,
                "LAnklePower"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kineticStats,
                "RAnklePower"+suffixPlus,"Right",2, color="blue", customLimits=None)

#
    def plotPanel(self):

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            self.fig.axes[0].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Moment"]["mean"][:,0]-self.m_normativeData["Hip.Moment"]["sd"][:,0],
                self.m_normativeData["Hip.Moment"]["mean"][:,0]+self.m_normativeData["Hip.Moment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[1].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Moment"]["mean"][:,1]-self.m_normativeData["Hip.Moment"]["sd"][:,1],
                self.m_normativeData["Hip.Moment"]["mean"][:,1]+self.m_normativeData["Hip.Moment"]["sd"][:,1],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[3].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Hip.Power"]["mean"][:,2]-self.m_normativeData["Hip.Power"]["sd"][:,2],
                self.m_normativeData["Hip.Power"]["mean"][:,2]+self.m_normativeData["Hip.Power"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)


            self.fig.axes[4].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Knee.Moment"]["mean"][:,0]-self.m_normativeData["Knee.Moment"]["sd"][:,0],
                self.m_normativeData["Knee.Moment"]["mean"][:,0]+self.m_normativeData["Knee.Moment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[5].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Knee.Moment"]["mean"][:,1]-self.m_normativeData["Knee.Moment"]["sd"][:,1],
                self.m_normativeData["Knee.Moment"]["mean"][:,1]+self.m_normativeData["Knee.Moment"]["sd"][:,1],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[7].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Knee.Power"]["mean"][:,2]-self.m_normativeData["Knee.Power"]["sd"][:,2],
                self.m_normativeData["Knee.Power"]["mean"][:,2]+self.m_normativeData["Knee.Power"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)



            self.fig.axes[8].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Ankle.Moment"]["mean"][:,0]-self.m_normativeData["Ankle.Moment"]["sd"][:,0],
                self.m_normativeData["Ankle.Moment"]["mean"][:,0]+self.m_normativeData["Ankle.Moment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[9].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Ankle.Moment"]["mean"][:,1]-self.m_normativeData["Ankle.Moment"]["sd"][:,1],
                self.m_normativeData["Ankle.Moment"]["mean"][:,1]+self.m_normativeData["Ankle.Moment"]["sd"][:,1],
                facecolor="green", alpha=0.5,linewidth=0)


            self.fig.axes[11].fill_between(np.linspace(0,100,51),
                self.m_normativeData["Ankle.Power"]["mean"][:,2]-self.m_normativeData["Ankle.Power"]["sd"][:,2],
                self.m_normativeData["Ankle.Power"]["mean"][:,2]+self.m_normativeData["Ankle.Power"]["sd"][:,2],
                facecolor="green", alpha=0.5,linewidth=0)

        return self.fig


class TemporalGaitKinematicsPlotViewer(AbstractPlotViewer):
    """
        **Description :** Constructor of temporal gait plot panel.

        .. note::

            The kinematic panel is made of 12 subplots

            ================  ==============================  =======  ===========
            matplotlib Axis   model outputs                   context  Axis label
            ================  ==============================  =======  ===========
            ax0               "LPelvisProgressAngles"         left     Tilt
                              "RPelvisProgressAngles"         right
            ax1               "LPelvisProgress.Angles"        left     Obli
                              "RPelvisProgress.Angles"        right
            ax2               "LPelvisProgress.Angles"        left     Rota
                              "RPelvisProgress.Angles"        right
            ax3               "LHipAngles"                    left     Flex
                              "RHipAngles"                    right
            ax4               "LHipAngles"                    left     Addu
                              "RHipAngles"                    right
            ax5               "LHipAngles"                    left    Rota
                              "RHipAngles"                    right
            ax6               "LKneeAngles"                   left     Flex
                              "RKneeAngles"                   right
            ax7               "LKneeAngles"                   left     Addu
                              "RKneeAngles"                   right
            ax8               "LKneeAngles"                   left     Rota
                              "RKneeAngles"                   right
            ax9              "LAnkleAngles"                  left     Flex
                              "RAnkleAngles"                  right
            ax10              "LAnkleAngles"                  left     Eve
                              "RAnkleAngles"                  right
            ax11              "LFootProgressAngles"           left
                              "RFootProgressAngles"           right
            ================  ==============================  =======  ===========

    """

    def __init__(self,iTrial,pointLabelSuffix=None):

        """
            :Parameters:
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
        """


        super(TemporalGaitKinematicsPlotViewer, self).__init__(iTrial)

        self.m_trial = self.m_input
        if isinstance(self.m_input,ma.Trial):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a ma.Trial")


        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Time Kinematics \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(5,3,1)# Pelvis X
        ax1 = plt.subplot(5,3,2)# Pelvis Y
        ax2 = plt.subplot(5,3,3)# Pelvis Z
        ax3 = plt.subplot(5,3,4)# Hip X
        ax4 = plt.subplot(5,3,5)# Hip Y
        ax5 = plt.subplot(5,3,6)# Hip Z
        ax6 = plt.subplot(5,3,7)# Knee X
        ax7 = plt.subplot(5,3,8)# Knee Y
        ax8 = plt.subplot(5,3,9)# Knee Z
        ax9 = plt.subplot(5,3,10)# Ankle X
        ax10 = plt.subplot(5,3,11)# Ankle Z
        ax11 = plt.subplot(5,3,12)# Footprogress Z

        ax0.set_title("Pelvis Tilt" ,size=8)
        ax1.set_title("Pelvis Obliquity" ,size=8)
        ax2.set_title("Pelvis Rotation" ,size=8)
        ax3.set_title("Hip Flexion" ,size=8)
        ax4.set_title("Hip Adduction" ,size=8)
        ax5.set_title("Hip Rotation" ,size=8)
        ax6.set_title("Knee Flexion" ,size=8)
        ax7.set_title("Knee Adduction" ,size=8)
        ax8.set_title("Knee Rotation" ,size=8)
        ax9.set_title("Ankle dorsiflexion" ,size=8)
        ax10.set_title("Ankle supination" ,size=8)
        ax11.set_title("Foot Progression " ,size=8)

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)


        ax9.set_xlabel("Frame",size=8)
        ax10.set_xlabel("Frame",size=8)
        ax11.set_xlabel("Frame",size=8)

        ax0.set_ylim([0,60])
        ax1.set_ylim([-30,30])
        ax2.set_ylim([-30,30])

        ax3.set_ylim( [-20,70])
        ax4.set_ylim([-30,30])
        ax5.set_ylim([-30,30])

        ax6.set_ylim([-15,75])
        ax7.set_ylim([-30,30])
        ax8.set_ylim([-30,30])

        ax9.set_ylim([-30,30])
        ax10.set_ylim([-30,30])
        ax11.set_ylim([-30,30])

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        pass

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        plot.temporalPlot(self.fig.axes[0],self.m_trial,
                                "LPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[1],self.m_trial,
                        "LPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[2],self.m_trial,
                                "LPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[3],self.m_trial,
                                "LHipAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[4],self.m_trial,
                                "LHipAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[5],self.m_trial,
                                "LHipAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[6],self.m_trial,
                                "LKneeAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[7],self.m_trial,
                                "LKneeAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[8],self.m_trial,
                                "LKneeAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[9],self.m_trial,
                                "LAnkleAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[10],self.m_trial,
                                "LAnkleAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[11],self.m_trial,
                                "LFootProgressAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")


        plot.temporalPlot(self.fig.axes[0],self.m_trial,
                                "RPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[1],self.m_trial,
                        "RPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[2],self.m_trial,
                                "RPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[3],self.m_trial,
                                "RHipAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[4],self.m_trial,
                                "RHipAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[5],self.m_trial,
                                "RHipAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[6],self.m_trial,
                                "RKneeAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[7],self.m_trial,
                                "RKneeAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[8],self.m_trial,
                                "RKneeAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[9],self.m_trial,
                                "RAnkleAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[10],self.m_trial,
                                "RAnkleAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[11],self.m_trial,
                                "RFootProgressAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

    def plotPanel(self):

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig


class TemporalGaitKineticsPlotViewer(AbstractPlotViewer):
    """
        **Description :** Constructor of temporal gait plot panel.

        .. note::

            The kinematic panel is made of 12 subplots

            ================  ==============================  =======  ===========
            matplotlib Axis   model outputs                   context  Axis label
            ================  ==============================  =======  ===========
            ax0               "LPelvisProgressAngles"         left     Tilt
                              "RPelvisProgressAngles"         right
            ax1               "LPelvisProgress.Angles"        left     Obli
                              "RPelvisProgress.Angles"        right
            ax2               "LPelvisProgress.Angles"        left     Rota
                              "RPelvisProgress.Angles"        right
            ax3               "LHipAngles"                    left     Flex
                              "RHipAngles"                    right
            ax4               "LHipAngles"                    left     Addu
                              "RHipAngles"                    right
            ax5               "LHipAngles"                    left    Rota
                              "RHipAngles"                    right
            ax6               "LKneeAngles"                   left     Flex
                              "RKneeAngles"                   right
            ax7               "LKneeAngles"                   left     Addu
                              "RKneeAngles"                   right
            ax8               "LKneeAngles"                   left     Rota
                              "RKneeAngles"                   right
            ax9              "LAnkleAngles"                  left     Flex
                              "RAnkleAngles"                  right
            ax10              "LAnkleAngles"                  left     Eve
                              "RAnkleAngles"                  right
            ax11              "LFootProgressAngles"           left
                              "RFootProgressAngles"           right
            ================  ==============================  =======  ===========

    """

    def __init__(self,iTrial,pointLabelSuffix=None):

        """
            :Parameters:
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
        """


        super(TemporalGaitKineticsPlotViewer, self).__init__(iTrial)

        self.m_trial = self.m_input
        if isinstance(self.m_input,ma.Trial):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a ma.Trial")

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Descriptive Time-normalized Kinetics \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(3,4,1)# Hip X extensor
        ax1 = plt.subplot(3,4,2)# Hip Y abductor
        ax2 = plt.subplot(3,4,3)# Hip Z rotation
        ax3 = plt.subplot(3,4,4)# Knee Z power

        ax4 = plt.subplot(3,4,5)# knee X extensor
        ax5 = plt.subplot(3,4,6)# knee Y abductor
        ax6 = plt.subplot(3,4,7)# knee Z rotation
        ax7 = plt.subplot(3,4,8)# knee Z power

        ax8 = plt.subplot(3,4,9)# ankle X plantar flexion
        ax9 = plt.subplot(3,4,10)# ankle Y rotation
        ax10 = plt.subplot(3,4,11)# ankle Z everter
        ax11 = plt.subplot(3,4,12)# ankle Z power

        ax0.set_title("Hip extensor Moment" ,size=8)
        ax1.set_title("Hip abductor Moment" ,size=8)
        ax2.set_title("Hip rotation Moment" ,size=8)
        ax3.set_title("Hip Power" ,size=8)

        ax4.set_title("Knee extensor Moment" ,size=8)
        ax5.set_title("Knee abductor Moment" ,size=8)
        ax6.set_title("Knee rotation Moment" ,size=8)
        ax7.set_title("Knee Power" ,size=8)

        ax8.set_title("Ankle plantarflexor Moment" ,size=8)
        ax9.set_title("Ankle everter Moment" ,size=8)
        ax10.set_title("Ankle abductor Moment" ,size=8)
        ax11.set_title("Ankle Power " ,size=8)

        for ax in [self.fig.axes[0],self.fig.axes[1],self.fig.axes[2],
                   self.fig.axes[4],self.fig.axes[5],self.fig.axes[0],
                   self.fig.axes[8],self.fig.axes[9],self.fig.axes[10]]:
            ax.set_ylabel("moment (N.mm.kg-1)",size=8)

        for ax in [self.fig.axes[3],self.fig.axes[7],self.fig.axes[8]]:
            ax.set_ylabel("power (W.Kg-1)",size=8)

        ax9.set_xlabel("Frame %",size=8)
        ax10.set_xlabel("Frame %",size=8)
        ax11.set_xlabel("Frame %",size=8)

        ax0.set_ylim([-2.0 *1000.0, 3.0*1000.0])
        ax1.set_ylim([-2.0*1000.0, 1.0*1000.0])
        ax2.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax3.set_ylim(  [-3.0, 3.0])

        ax4.set_ylim([-1.0*1000.0, 1.0*1000.0])
        ax5.set_ylim([-1.0*1000.0, 1.0*1000.0])
        ax6.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax7.set_ylim( [-3.0, 3.0])

        ax8.set_ylim([-1.0*1000.0, 3.0*1000.0])
        ax9.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax10.set_ylim([-0.5*1000.0, 0.5*1000.0])
        ax11.set_ylim( [-2.0, 5.0])

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        pass

    def __setData(self):


        plot.temporalPlot(self.fig.axes[0],self.m_trial,
                                "LHipMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[1],self.m_trial,
                        "LHipMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[2],self.m_trial,
                                "LHipMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[3],self.m_trial,
                                "LHipPower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")

        plot.temporalPlot(self.fig.axes[4],self.m_trial,
                                "LKneeMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[5],self.m_trial,
                                "LKneeMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[6],self.m_trial,
                                "LKneeMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[7],self.m_trial,
                                "LKneePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")

        plot.temporalPlot(self.fig.axes[8],self.m_trial,
                                "LAnkleMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[9],self.m_trial,
                                "LAnkleMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[10],self.m_trial,
                                "LAnkleMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
        plot.temporalPlot(self.fig.axes[11],self.m_trial,
                                "LAnklePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")


        plot.temporalPlot(self.fig.axes[0],self.m_trial,
                                "RHipMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[1],self.m_trial,
                        "RHipMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[2],self.m_trial,
                                "RHipMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[3],self.m_trial,
                                "RHipPower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

        plot.temporalPlot(self.fig.axes[4],self.m_trial,
                                "RKneeMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[5],self.m_trial,
                                "RKneeMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[6],self.m_trial,
                                "RKneeMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[7],self.m_trial,
                                "RKneePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

        plot.temporalPlot(self.fig.axes[8],self.m_trial,
                                "RAnkleMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[9],self.m_trial,
                                "RAnkleMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[10],self.m_trial,
                                "RAnkleMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
        plot.temporalPlot(self.fig.axes[11],self.m_trial,
                                "RAnklePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")




    def plotPanel(self):

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig
