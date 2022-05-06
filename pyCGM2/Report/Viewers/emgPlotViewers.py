# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

"""
Module contains `plotViewers` for displaying emg data
"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

import matplotlib.pyplot as plt

# pyCGM2
import pyCGM2
from pyCGM2.Report import plot
from pyCGM2.Report import plotUtils
from pyCGM2.Report.Viewers import plotViewers



class TemporalEmgPlotViewer(plotViewers.AbstractPlotViewer):
    """plot temporal emg plot

    Args:
        iAcq (btk.Acquisition): an acquisition
        pointLabelSuffix (str,Optional[None]): suffix added to emg outputs


    """

    def __init__(self,iAcq,pointLabelSuffix=None):


        super(TemporalEmgPlotViewer, self).__init__(iAcq)

        self.emgs = list()
        self.rectify = False

        self.m_acq = self.m_input
        if isinstance(self.m_input,pyCGM2.btk.btkAcquisition):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a ma.Trial")

        self.m_pointLabelSuffix = pointLabelSuffix

        self.m_ignoreNormalActivity = False
        self.m_selectChannels = None

    def setEmgManager(self,emgManager):
        """set the `emgManager` instance

        Args:
            emgManager (pyCGM2.EMG.EmgManager): `emgManager` instance

        """

        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()

    def selectEmgChannels(self,channelNames):
        """set the emg channels

        Args:
            channelNames (str): channel labels

        """
        self.m_selectChannels = channelNames

    def setEmgRectify(self, flag):
        """Enable/disable rectify mode

        Args:
            flag (bool): boolean flag

        """

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


        i=0
        for channel in self.m_selectChannels:
            label = channel
            context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]
            muscle = self.m_emgmanager.m_emgChannelSection[channel]["Muscle"]
            normalActivity = self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] if self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] is not None else ""

            self.fig.axes[i].set_title(label +":"+ muscle+"-"+context+"["+ normalActivity+"]" ,size=6)
            i+=1


        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)


    def setNormativeDataset(self,iNormativeDataSet):
        pass


    def ignoreNormalActivty(self, bool):
        """Enable/disable normal actividy display

        Args:
            bool (bool): boolean flag

        """

        self.m_ignoreNormalActivity = bool

    def __setData(self):

        i=0
        for channel in self.m_selectChannels:
            label = channel+"_Rectify" if self.rectify  else channel+"_HPF"
            context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]
            colorContext = plotUtils.colorContext(context)

            normalActivationLabel = self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] if self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] is not None else ""

            plot.temporalPlot(self.fig.axes[i],self.m_acq,
                                    label,0,
                                    color=colorContext,linewidth=1)

            if not self.m_ignoreNormalActivity:
                plot.addTemporalNormalActivationLayer(self.fig.axes[i],self.m_acq,normalActivationLabel,context)
            i+=1


    def plotPanel(self):
        """ plot the panel
        """

        self.__setLayer()
        self.__setData()

        return self.fig


class CoactivationEmgPlotViewer(plotViewers.AbstractPlotViewer):
    """plot coactivation plot

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        pointLabelSuffix (str,Optional[None]): suffix added to emg outputs


    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):

        super(CoactivationEmgPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

    def setEmgs(self,label1,label2):
        """set the 2 emg labels to plot

        Args:
            label1 (str): emg channel label
            label2 (str): emg channel label

        """
        self.m_emg1 = label1+"_Rectify_Env_Norm"
        self.m_emg2 = label2+"_Rectify_Env_Norm"

    def setMuscles(self,label1,label2):
        """set the 2 measured muscle names

        Args:
            label1 (str): muscle name
            label2 (str): muscle name

        """
        self.m_muscleLabel1 = label1
        self.m_muscleLabel2 = label2

    def setContext(self,context):
        """set event context

        Args:
            context (str): event context

        """
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
        """set a concrete plot function

        Args:
            concreteplotFunction (pyCGM2.Report.plot): plot function

        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):

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


    def plotPanel(self):
        """plot the panel"""

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig

class EnvEmgGaitPlotPanelViewer(plotViewers.AbstractPlotViewer):
    """plot emg envelops

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        pointLabelSuffix (str,Optional[None]): suffix added to emg outputs


    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):

        super(EnvEmgGaitPlotPanelViewer, self).__init__(iAnalysis)

        self.emgs = list()
        self.m_normalActivEmgs = list()

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

        self.m_normalizedEmgFlag = False
        self.m_selectChannels = None

    def selectEmgChannels(self,channelNames):
        """set the emg channels

        Args:
            channelNames (str): channel labels

        """
        self.m_selectChannels = channelNames

    def setEmgManager(self,emgManager):
        """set the `emgManager` instance

        Args:
            emgManager (pyCGM2.EMG.EmgManager): `emgManager` instance

        """
        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()


    def setNormalizedEmgFlag(self,flag):
        """Enable/Disable amplitude-normalized emg

        Args:
            flag (bool): boolean flag

        """
        self.m_normalizedEmgFlag = flag


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" EMG Envelop \n """ if not self.m_normalizedEmgFlag  else u""" Normalized EMG Envelop \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(4,4,1)
        ax1 = plt.subplot(4,4,2)
        ax2 = plt.subplot(4,4,3)
        ax3 = plt.subplot(4,4,4)
        ax4 = plt.subplot(4,4,5)
        ax5 = plt.subplot(4,4,6)
        ax6 = plt.subplot(4,4,7)
        ax7 = plt.subplot(4,4,8)
        ax8 = plt.subplot(4,4,9)
        ax9 = plt.subplot(4,4,10)
        ax10 = plt.subplot(4,4,11)
        ax11 = plt.subplot(4,4,12)
        ax12 = plt.subplot(4,4,13)
        ax13 = plt.subplot(4,4,14)
        ax14 = plt.subplot(4,4,15)
        ax15 = plt.subplot(4,4,16)

        i=0
        for channel in self.m_selectChannels:
            label = channel
            context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]
            muscle = self.m_emgmanager.m_emgChannelSection[channel]["Muscle"]
            normalActivity = self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] if self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] is not None else ""

            self.fig.axes[i].set_title(label+":"+ muscle+"-"+context+"\n["+ normalActivity+"]" ,size=6)
            i+=1

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)



    def setNormativeDataset(self,iNormativeDataSet):
        pass

    def setConcretePlotFunction(self, concreteplotFunction):
        """set a concrete plot function

        Args:
            concreteplotFunction (pyCGM2.Report.plot): plot function

        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):

        i=0
        for channel in self.m_selectChannels:
            label = channel+"_Rectify_Env" if not self.m_normalizedEmgFlag else channel+"_Rectify_Env_Norm"
            context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]
            colorContext = plotUtils.colorContext(context)

            normalActivationLabel = self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] if self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] is not None else ""


            self.m_concretePlotFunction(self.fig.axes[i],self.m_analysis.emgStats,
                            label,context,0,color=colorContext)

            footOff = self.m_analysis.emgStats.pst['stancePhase', context]["mean"]
            plot.addNormalActivationLayer(self.fig.axes[i],normalActivationLabel, footOff)

            i+=1

    def plotPanel(self):
        """plot the panel"""

        self.__setLayer()
        self.__setData()

        return self.fig



class MultipleAnalysis_EnvEmgPlotPanelViewer(plotViewers.AbstractPlotViewer):
    """plot emg envelops from multiple `analysis` instances

    Args:
        iAnalyses (list):  `analysis` instances
        legend(str): label assoaciated to each instance
        pointLabelSuffix (str,Optional[None]): suffix added to emg outputs


    """

    def __init__(self,iAnalyses,legends,pointLabelSuffix=None):


        super(MultipleAnalysis_EnvEmgPlotPanelViewer, self).__init__(iAnalyses)


        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")


        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input


        self.emgs = list()
        self.m_normalActivEmgs = list()


        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normalizedEmgFlag = False
        self.m_legends = legends
        self.m_selectChannels = None

    def selectEmgChannels(self,channelNames):
        """set the emg channels

        Args:
            channelNames (str): channel labels

        """
        self.m_selectChannels = channelNames

    def setEmgManager(self,emgManager):
        """set the `emgManager` instance

        Args:
            emgManager (pyCGM2.EMG.EmgManager): `emgManager` instance

        """
        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()

    def setNormalizedEmgFlag(self,flag):
        """Enable/Disable amplitude-normalized emg

        Args:
            flag (bool): boolean flag

        """
        self.m_normalizedEmgFlag = flag


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" EMG Envelop \n """ if not self.m_normalizedEmgFlag  else u""" Normalized EMG Envelop \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(4,4,1)
        ax1 = plt.subplot(4,4,2)
        ax2 = plt.subplot(4,4,3)
        ax3 = plt.subplot(4,4,4)
        ax4 = plt.subplot(4,4,5)
        ax5 = plt.subplot(4,4,6)
        ax6 = plt.subplot(4,4,7)
        ax7 = plt.subplot(4,4,8)
        ax8 = plt.subplot(4,4,9)
        ax9 = plt.subplot(4,4,10)
        ax10 = plt.subplot(4,4,11)
        ax11 = plt.subplot(4,4,12)
        ax12 = plt.subplot(4,4,13)
        ax13 = plt.subplot(4,4,14)
        ax14 = plt.subplot(4,4,15)
        ax15 = plt.subplot(4,4,16)

        i=0
        for channel in self.m_selectChannels:
            label = channel
            context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]
            muscle = self.m_emgmanager.m_emgChannelSection[channel]["Muscle"]
            normalActivity = self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] if self.m_emgmanager.m_emgChannelSection[channel]["NormalActivity"] is not None else ""

            self.fig.axes[i].set_title(label+":"+ muscle+"-"+context+"\n["+ normalActivity+"]" ,size=6)

            i+=1

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)

    def __setLegend(self,axisIndex):
        self.fig.axes[axisIndex].legend(fontsize=6)


    def setNormativeDataset(self,iNormativeDataSet):
        pass

    def setConcretePlotFunction(self, concreteplotFunction):
        """set a concrete plot function

        Args:
            concreteplotFunction (pyCGM2.Report.plot): plot function

        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        #suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

            colormap_i_left=[plt.cm.Reds(k) for k in np.linspace(0.2, 1, len(self.m_analysis))]
            colormap_i_right=[plt.cm.Blues(k) for k in np.linspace(0.2, 1, len(self.m_analysis))]


            j = 0
            for analysis in self.m_analysis:

                i=0
                for channel in self.m_selectChannels:
                    label = channel+"_Rectify_Env" if not self.m_normalizedEmgFlag else channel+"_Rectify_Env_Norm"
                    context = self.m_emgmanager.m_emgChannelSection[channel]["Context"]

                    if context == "Left":
                        self.m_concretePlotFunction(self.fig.axes[i],analysis.emgStats,
                                        label,context,0,color=colormap_i_left[j],legendLabel=self.m_legends[j])
                    elif context =="Right":
                        self.m_concretePlotFunction(self.fig.axes[i],analysis.emgStats,
                                        label,context,0,color=colormap_i_right[j],legendLabel=self.m_legends[j])
                    i+=1

                j+=1



    def plotPanel(self):
        """plot the panel"""


        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

        # normative dataset not implemented

        return self.fig
