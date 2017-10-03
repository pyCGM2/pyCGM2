# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# pyCGM2
import pyCGM2
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Report import plot

# openMA
import ma.io
import ma.body

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

class GaitKinematicsPlotViewer(AbstractPlotViewer):
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
            ================  ==============================  =======  ===========

    """

    def __init__(self,iAnalysis,pointLabelSuffix="",plotType=pyCGM2Enums.PlotType.DESCRIPTIVE):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(GaitKinematicsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")


        self.m_type = plotType
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Descriptive Time-normalized Kinematics \n """
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
        ax10.set_title("Ankle eversion" ,size=8)
        ax11.set_title("Foot Progression " ,size=8)

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)

        ax9.set_xlabel("Gait cycle %",size=8)
        ax10.set_xlabel("Gait cycle %",size=8)
        ax11.set_xlabel("Gait cycle %",size=8)

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
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

        if self.m_type == pyCGM2Enums.PlotType.DESCRIPTIVE:
            # pelvis
            plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # hip
            plot.gaitDescriptivePlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitDescriptivePlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            # foot progress
            plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "LFootProgressAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "RFootProgressAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

        elif self.m_type == pyCGM2Enums.PlotType.CONSISTENCY:
            # pelvis
            plot.gaitConsistencyPlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # hip
            plot.gaitConsistencyPlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitConsistencyPlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitConsistencyPlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            # foot progress
            plot.gaitConsistencyPlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "LFootProgressAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "RFootProgressAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

        elif self.m_type == pyCGM2Enums.PlotType.MEAN_ONLY:
            # pelvis
            plot.gaitMeanPlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "LPelvisAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "RPelvisAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # hip
            plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "LHipAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "RHipAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitMeanPlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "LKneeAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "RKneeAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitMeanPlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "LAnkleAngles"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "RAnkleAngles"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            # foot progress
            plot.gaitMeanPlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "LFootProgressAngles"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "RFootProgressAngles"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)
#
    def plotPanel(self):

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

class GaitKineticsPlotViewer(AbstractPlotViewer):
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

    def __init__(self,iAnalysis,pointLabelSuffix="",plotType=pyCGM2Enums.PlotType.DESCRIPTIVE):

        """
            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(GaitKineticsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")


        self.m_type = plotType
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False

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

        ax9.set_xlabel("Gait cycle %",size=8)
        ax10.set_xlabel("Gait cycle %",size=8)
        ax11.set_xlabel("Gait cycle %",size=8)

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
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

        if self.m_type == pyCGM2Enums.PlotType.DESCRIPTIVE:
            # hip
            plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "LHipPower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "RHipPower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitDescriptivePlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "LKneePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "RKneePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitDescriptivePlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "LAnklePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "RAnklePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

        elif self.m_type == pyCGM2Enums.PlotType.CONSISTENCY:
            # hip
            plot.gaitConsistencyPlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "LHipPower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "RHipPower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitConsistencyPlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "LKneePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "RKneePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitConsistencyPlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitConsistencyPlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "LAnklePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitConsistencyPlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "RAnklePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

        elif self.m_type == pyCGM2Enums.PlotType.MEAN_ONLY:
            # hip
            plot.gaitMeanPlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[0],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[1],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "LHipMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[2],self.m_analysis.kineticStats,
                    "RHipMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "LHipPower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.kineticStats,
                    "RHipPower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # knee
            plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "LKneeMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[6],self.m_analysis.kineticStats,
                    "RKneeMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "LKneePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.kineticStats,
                    "RKneePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            # ankle
            plot.gaitMeanPlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",0, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[8],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",0, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",1, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[9],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",1, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "LAnkleMoment"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[10],self.m_analysis.kineticStats,
                    "RAnkleMoment"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)

            plot.gaitMeanPlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "LAnklePower"+suffixPlus,"Left",2, color="red", addPhaseFlag=True,customLimits=None)
            plot.gaitMeanPlot(self.fig.axes[11],self.m_analysis.kineticStats,
                    "RAnklePower"+suffixPlus,"Right",2, color="blue", addPhaseFlag=True,customLimits=None)
#
    def plotPanel(self):

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

class multipleAnalyses_GaitKinematicsPlotViewer(AbstractPlotViewer):
    """


    """

    def __init__(self,iAnalyses,context,legends,pointLabelSuffix_lst=None,
                plotType=pyCGM2Enums.PlotType.CONSISTENCY):

        """
            :Parameters:
                 - `iAnalyses` (list ) - list of ` of pyCGM2.Processing.analysis.Analysis`
                 - `context` (str ) - list of ` of pyCGM2.Processing.analysis.Analysis`
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(multipleAnalyses_GaitKinematicsPlotViewer, self).__init__(iAnalyses)


        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")


        self.m_context = context
        self.m_pointLabelSuffixes = pointLabelSuffix_lst
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_legends = legends
        self.m_type = plotType

        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")
        if pointLabelSuffix_lst is not None:
            if len(iAnalyses) != len(pointLabelSuffix_lst):
                raise Exception("list of point label suffix don t match analysis. Must have same length")



    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Descriptive Time-normalized Kinematics \n """
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
        ax10.set_title("Ankle eversion" ,size=8)
        ax11.set_title("Foot Progression " ,size=8)

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)


        ax9.set_xlabel("Gait cycle %",size=8)
        ax10.set_xlabel("Gait cycle %",size=8)
        ax11.set_xlabel("Gait cycle %",size=8)

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


    def __setLegend(self,axisIndex):
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):

        if self.m_type == pyCGM2Enums.PlotType.CONSISTENCY:

            if self.m_context == "Left":
                colormap = plt.cm.Reds
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    plot.gaitConsistencyPlot(self.fig.axes[0],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                    legendLabel=legend)


                    plot.gaitConsistencyPlot(self.fig.axes[1],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[2],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # hip
                    plot.gaitConsistencyPlot(self.fig.axes[3],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[4],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[5],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitConsistencyPlot(self.fig.axes[6],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[7],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[8],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitConsistencyPlot(self.fig.axes[9],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[10],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # foot progress
                    plot.gaitConsistencyPlot(self.fig.axes[11],analysis.kinematicStats,
                            "LFootProgressAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
#
                    i+=1
            if self.m_context == "Right":
                colormap = plt.cm.Blues
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    plot.gaitConsistencyPlot(self.fig.axes[0],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                    legendLabel=legend)


                    plot.gaitConsistencyPlot(self.fig.axes[1],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[2],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # hip
                    plot.gaitConsistencyPlot(self.fig.axes[3],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[4],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[5],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitConsistencyPlot(self.fig.axes[6],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[7],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[8],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitConsistencyPlot(self.fig.axes[9],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitConsistencyPlot(self.fig.axes[10],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # foot progress
                    plot.gaitConsistencyPlot(self.fig.axes[11],analysis.kinematicStats,
                            "RFootProgressAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
            #
                    i+=1

        if self.m_type == pyCGM2Enums.PlotType.MEAN_ONLY:

            if self.m_context == "Left":
                colormap = plt.cm.Reds
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    plot.gaitMeanPlot(self.fig.axes[0],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                    legendLabel=legend)


                    plot.gaitMeanPlot(self.fig.axes[1],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[2],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # hip
                    plot.gaitMeanPlot(self.fig.axes[3],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[4],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[5],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitMeanPlot(self.fig.axes[6],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[7],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[8],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitMeanPlot(self.fig.axes[9],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[10],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # foot progress
                    plot.gaitMeanPlot(self.fig.axes[11],analysis.kinematicStats,
                            "LFootProgressAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
#
                    i+=1
            if self.m_context == "Right":
                colormap = plt.cm.Blues
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    plot.gaitMeanPlot(self.fig.axes[0],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                    legendLabel=legend)


                    plot.gaitMeanPlot(self.fig.axes[1],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[2],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # hip
                    plot.gaitMeanPlot(self.fig.axes[3],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[4],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[5],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitMeanPlot(self.fig.axes[6],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[7],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[8],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitMeanPlot(self.fig.axes[9],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    plot.gaitMeanPlot(self.fig.axes[10],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # foot progress
                    plot.gaitMeanPlot(self.fig.axes[11],analysis.kinematicStats,
                            "RFootProgressAngles"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
            #
                    i+=1

#
    def plotPanel(self):

        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

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

class multipleAnalyses_GaitKineticsPlotViewer(AbstractPlotViewer):
    """


    """

    def __init__(self,iAnalyses,context,legends,pointLabelSuffix_lst=None,
                plotType=pyCGM2Enums.PlotType.CONSISTENCY):

        """
            :Parameters:
                 - `iAnalyses` (list ) - list of ` of pyCGM2.Processing.analysis.Analysis`
                 - `context` (str ) - list of ` of pyCGM2.Processing.analysis.Analysis`
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
    """


        super(multipleAnalyses_GaitKineticsPlotViewer, self).__init__(iAnalyses)

        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input
        self.m_context = context
        self.m_pointLabelSuffixes = pointLabelSuffix_lst
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_legends = legends
        self.m_type = plotType

        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")
        if pointLabelSuffix_lst is not None:
            if len(iAnalyses) != len(pointLabelSuffix_lst):
                raise Exception("list of point label suffix don t match analysis. Must have same length")



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

        ax9.set_xlabel("Gait cycle %",size=8)
        ax10.set_xlabel("Gait cycle %",size=8)
        ax11.set_xlabel("Gait cycle %",size=8)

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

    def __setLegend(self,axisIndex):
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `iNormativeDataSet` (a class of the pyCGM2.Report.normativeDataset module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """
        iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):

        if self.m_type == pyCGM2Enums.PlotType.CONSISTENCY:

            if self.m_context == "Left":
                colormap = plt.cm.Reds
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    # hip
                    plot.gaitConsistencyPlot(self.fig.axes[0],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                            legendLabel=legend)

                    plot.gaitConsistencyPlot(self.fig.axes[1],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)


                    plot.gaitConsistencyPlot(self.fig.axes[2],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[3],analysis.kineticStats,
                            "LHipPower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitConsistencyPlot(self.fig.axes[4],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[5],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[6],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[7],analysis.kineticStats,
                            "LKneePower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitConsistencyPlot(self.fig.axes[8],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[9],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[10],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[11],analysis.kineticStats,
                            "LAnklePower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)
                    i+=1

            if self.m_context == "Right":
                colormap = plt.cm.Blues
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    # hip
                    plot.gaitConsistencyPlot(self.fig.axes[0],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                            legendLabel=legend)

                    plot.gaitConsistencyPlot(self.fig.axes[1],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)


                    plot.gaitConsistencyPlot(self.fig.axes[2],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[3],analysis.kineticStats,
                            "RHipPower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitConsistencyPlot(self.fig.axes[4],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[5],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[6],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[7],analysis.kineticStats,
                            "RKneePower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitConsistencyPlot(self.fig.axes[8],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[9],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[10],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitConsistencyPlot(self.fig.axes[11],analysis.kineticStats,
                            "RAnklePower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    i+=1

        if self.m_type == pyCGM2Enums.PlotType.MEAN_ONLY:

            if self.m_context == "Left":
                colormap = plt.cm.Reds
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    # hip
                    plot.gaitMeanPlot(self.fig.axes[0],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                            legendLabel=legend)

                    plot.gaitMeanPlot(self.fig.axes[1],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)


                    plot.gaitMeanPlot(self.fig.axes[2],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[3],analysis.kineticStats,
                            "LHipPower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitMeanPlot(self.fig.axes[4],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[5],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[6],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[7],analysis.kineticStats,
                            "LKneePower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitMeanPlot(self.fig.axes[8],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[9],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[10],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[11],analysis.kineticStats,
                            "LAnklePower"+suffixPlus,"Left",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    i+=1


            if self.m_context == "Right":
                colormap = plt.cm.Blues
                colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]

                i = 0
                for analysis in self.m_analysis:

                    if self.m_pointLabelSuffixes is not None:
                        suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
                    else:
                        suffixPlus=""

                    legend= self.m_legends[i]

                    # hip
                    plot.gaitMeanPlot(self.fig.axes[0],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None,
                            legendLabel=legend)

                    plot.gaitMeanPlot(self.fig.axes[1],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)


                    plot.gaitMeanPlot(self.fig.axes[2],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[3],analysis.kineticStats,
                            "RHipPower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # knee
                    plot.gaitMeanPlot(self.fig.axes[4],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[5],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[6],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[7],analysis.kineticStats,
                            "RKneePower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    # ankle
                    plot.gaitMeanPlot(self.fig.axes[8],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",0, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[9],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",1, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[10],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    plot.gaitMeanPlot(self.fig.axes[11],analysis.kineticStats,
                            "RAnklePower"+suffixPlus,"Right",2, color=colormap_i[i], addPhaseFlag=True,customLimits=None)

                    i+=1
#
    def plotPanel(self):

        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

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

    def __init__(self,iTrial,pointLabelSuffix=""):

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
        ax10.set_title("Ankle eversion" ,size=8)
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
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

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

    def __init__(self,iTrial,pointLabelSuffix=""):

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
