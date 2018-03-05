# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# pyCGM2
import pyCGM2
from pyCGM2.Report import plot, plotViewers,plotUtils
from pyCGM2.EMG import normalActivation
from pyCGM2.Processing import cycle
# openMA
import ma.io
import ma.body



class TemporalEmgPlotViewer(plotViewers.AbstractPlotViewer):
    """

    """

    def __init__(self,iTrial,pointLabelSuffix=""):

        """
            :Parameters:
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
        """


        super(TemporalEmgPlotViewer, self).__init__(iTrial)

        self.emgs = list()
        self.rectify = False

        self.m_trial = self.m_input
        if isinstance(self.m_input,ma.Trial):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a ma.Trial")

        self.m_pointLabelSuffix = pointLabelSuffix

    def setEmgs(self,emgs):
        for it in emgs:
            self.emgs.append({"Label": it[0], "Context": it[1]})

    def setEmgRectify(self, flag):
        self.rectify = flag

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Temporal EMG \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)

        ax0 = plt.subplot(10,1,1)
        ax1 = plt.subplot(10,1,2)
        ax2 = plt.subplot(10,1,3)
        ax3 = plt.subplot(10,1,4)
        ax4 = plt.subplot(10,1,5)
        ax5 = plt.subplot(10,1,6)
        ax6 = plt.subplot(10,1,7)
        ax7 = plt.subplot(10,1,8)
        ax8 = plt.subplot(10,1,9)
        ax9 = plt.subplot(10,1,10)

        for i in range(0, len(self.emgs)):
            label = self.emgs[i]["Label"]
            context = self.emgs[i]["Context"]

            muscle = self.m_normalActivEmgs[i] if self.m_normalActivEmgs[i] is not None else ""

            self.fig.axes[i].set_title(label +"["+ muscle+"]" +"-"+context ,size=6)


        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)


    def setNormativeDataset(self,iNormativeDataSet):
        pass

    def setNormalActivationLabels(self, labels):
        self.m_normalActivEmgs = labels


    def __setData(self):
        #suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

        for i in range(0, len(self.emgs)):
            label = self.emgs[i]["Label"]+"_Rectify" if self.rectify  else self.emgs[i]["Label"]+"_HPF"
            context = self.emgs[i]["Context"]
            colorContext = plotUtils.colorContext(context)

            normalActivationLabel = self.m_normalActivEmgs[i]

            plot.temporalPlot(self.fig.axes[i],self.m_trial,
                                    label,0,
                                    color=colorContext)
            plot.addTemporalNormalActivationLayer(self.fig.axes[i],self.m_trial,self.m_normalActivEmgs[i],context)


    def plotPanel(self):

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig


class CoactivationEmgPlotViewer(plotViewers.AbstractPlotViewer):
    """

    """

    def __init__(self,iAnalysis,pointLabelSuffix=""):

        """
            :Parameters:
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
        """


        super(CoactivationEmgPlotViewer, self).__init__(iAnalysis)


        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

    def setEmgs(self,label1,label2):
        self.m_emg1 = label1+"_Rectify_Env_Norm"
        self.m_emg2 = label2+"_Rectify_Env_Norm"

    def setMuscles(self,label1,label2):
        self.m_muscleLabel1 = label1
        self.m_muscleLabel2 = label2

    def setContext(self,context):
        self.m_context = context

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=" Coactivation : " + self.m_muscleLabel1 + " Vs " + self.m_muscleLabel2
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(1,1,1)


        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)


    def setNormativeDataset(self,iNormativeDataSet):
        pass

    def setConcretePlotFunction(self, concreteplotFunction):
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        #suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""


        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.emgStats,
                        self.m_emg1,self.m_context,0,color="red")
        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.emgStats,
                        self.m_emg2,self.m_context,0,color="blue")

        mean1=self.m_analysis.emgStats.data[self.m_emg1,self.m_context]["mean"][:,0]
        mean2=self.m_analysis.emgStats.data[self.m_emg2,self.m_context]["mean"][:,0]

        commonEmg=np.zeros(((101)))
        for i in range(0,101):
            commonEmg[i]=np.minimum(mean1[i],mean2[i])

        self.fig.axes[0].fill_between(np.arange(0,101,1), 0, commonEmg,facecolor='grey', alpha=0.7)
        #self.fig.axes[0].plot(np.arange(0,101,1), commonEmg, color='black')

    def plotPanel(self):

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig

class EnvEmgGaitPlotPanelViewer(plotViewers.AbstractPlotViewer):
    """

    """

    def __init__(self,iAnalysis,pointLabelSuffix=""):

        """
            :Parameters:
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
        """


        super(EnvEmgGaitPlotPanelViewer, self).__init__(iAnalysis)

        self.emgs = list()
        self.m_normalActivEmgs = list()

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

    def setEmgs(self,emgs):
        for it in emgs:
            self.emgs.append({"Label": it[0], "Context": it[1]})

    def setNormalActivationLabels(self, labels):
        self.m_normalActivEmgs = labels

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Normalized EMG \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(5,3,1)
        ax1 = plt.subplot(5,3,2)
        ax2 = plt.subplot(5,3,3)
        ax3 = plt.subplot(5,3,4)
        ax4 = plt.subplot(5,3,5)
        ax5 = plt.subplot(5,3,6)
        ax6 = plt.subplot(5,3,7)
        ax7 = plt.subplot(5,3,8)
        ax8 = plt.subplot(5,3,9)
        ax9 = plt.subplot(5,3,10)
        ax10 = plt.subplot(5,3,11)
        ax11 = plt.subplot(5,3,12)

        for i in range(0, len(self.emgs)):
            label = self.emgs[i]["Label"]
            context = self.emgs[i]["Context"]

            muscle = self.m_normalActivEmgs[i] if self.m_normalActivEmgs[i] is not None else ""

            self.fig.axes[i].set_title(label +"["+ muscle+"]" +"-"+context ,size=6)


        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)



    def setNormativeDataset(self,iNormativeDataSet):
        pass

    def setConcretePlotFunction(self, concreteplotFunction):
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        #suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""


        for i in range(0, len(self.emgs)):

            label = self.emgs[i]["Label"]+"_Rectify_Env_Norm"
            context = self.emgs[i]["Context"]
            colorContext = plotUtils.colorContext(context)

            self.m_concretePlotFunction(self.fig.axes[i],self.m_analysis.emgStats,
                            label,context,0,color=colorContext)

            footOff = self.m_analysis.emgStats.pst['stancePhase', context]["mean"]
            plot.addNormalActivationLayer(self.fig.axes[i],self.m_normalActivEmgs[i], footOff)


    def plotPanel(self):

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig
