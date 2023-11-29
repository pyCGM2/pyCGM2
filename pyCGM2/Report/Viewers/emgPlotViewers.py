
"""
Module contains `plotViewers` to display emg data
"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

import matplotlib.pyplot as plt

# pyCGM2
import pyCGM2
from pyCGM2.Report import plot
from pyCGM2.Report import plotUtils
from pyCGM2.Report.Viewers import plotViewers
from pyCGM2.EMG.emgManager import EmgManager
from pyCGM2.Report.normativeDatasets import NormativeData
from pyCGM2.Processing.analysis import Analysis
import btk

from typing import List, Tuple, Dict, Optional, Union, Callable

class TemporalEmgPlotViewer(plotViewers.PlotViewer):
    """
    A viewer for plotting temporal EMG data.

    This viewer plots raw or processed EMG data from a btk.Acquisition instance over time, 
    allowing for the visualization of EMG activity during the course of an acquisition.

    Args:
        iAcq (btk.Acquisition): The acquisition containing EMG data.
        pointLabelSuffix (str, optional): A suffix added to EMG channel names for custom labeling.
    """

    def __init__(self,iAcq:btk.btkAcquisition,pointLabelSuffix:Optional[str]=None):
        """Initialize the TemporalEmgPlotViewer."""

        super(TemporalEmgPlotViewer, self).__init__(iAcq)

        self.emgs = []
        self.rectify = False

        self.m_acq = self.m_input
        if isinstance(self.m_input,btk.btkAcquisition):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a ma.Trial")

        self.m_pointLabelSuffix = pointLabelSuffix

        self.m_ignoreNormalActivity = False
        self.m_selectChannels = None

    def setEmgManager(self,emgManager:EmgManager):
        """
        Set the EMG Manager for the viewer.

        Args:
            emgManager (EmgManager): An instance of EmgManager containing EMG processing details.
        """

        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()

    def selectEmgChannels(self,channelNames:List[str]):
        """
        Select specific EMG channels for plotting.

        Args:
            channelNames (List[str]): List of EMG channel names to be plotted.
        """
        self.m_selectChannels = channelNames

    def setEmgRectify(self, flag:bool):
        """
        Enable or disable EMG rectification in the plot.

        Args:
            flag (bool): True to enable rectification, False to disable.
        """

        self.rectify = flag

    def __setLayer(self):
        """
        Internal method to set up the plot layers.

        Configures the plot layout, titles, and axes for EMG visualization.
        """


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


    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Set a normative dataset for comparison.

        Note: Currently, displaying normal EMG data is not implemented.

        Args:
            iNormativeDataSet (NormativeData): Instance of NormativeData for comparison.
        """
        LOGGER.logger.warning("[pyCGM2] - pycgm2 not include and display normal emg data")
        pass


    def ignoreNormalActivty(self, bool:bool):
        """
        Enable or disable the display of normal EMG activity in the plot.

        Args:
            bool (bool): True to ignore normal EMG activity, False to display it.
        """

        self.m_ignoreNormalActivity = bool

    def __setData(self):
        """
        Internal method to set the data for the plot.

        Processes and plots the EMG data from the acquisition instance.
        """

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
        """
        Plot the panel with EMG data.

        Generates and returns the final plot figure with configured EMG data.
        """

        self.__setLayer()
        self.__setData()

        return self.fig


class CoactivationEmgPlotViewer(plotViewers.PlotViewer):
    """
    A viewer for plotting EMG coactivation data.

    This viewer facilitates the visualization of muscle coactivation by plotting EMG data 
    from two selected muscles and highlighting their coactivation periods.

    Args:
        iAnalysis (Analysis): An Analysis instance containing processed EMG data.
        pointLabelSuffix (str, optional): A suffix added to EMG outputs for custom labeling.
    """

    def __init__(self,iAnalysis:Analysis,pointLabelSuffix:Optional[str]=None):
        """Initialize the CoactivationEmgPlotViewer."""

        super(CoactivationEmgPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

    def setEmgs(self,label1:str,label2:str):
        """
        Set the EMG labels for coactivation analysis.

        Args:
            label1 (str): Label of the first EMG channel.
            label2 (str): Label of the second EMG channel.
        """
        self.m_emg1 = label1+"_Rectify_Env_Norm"
        self.m_emg2 = label2+"_Rectify_Env_Norm"

    def setMuscles(self,label1:str,label2:str):
        """
        Set the names of the muscles for coactivation analysis.

        Args:
            label1 (str): Name of the first muscle.
            label2 (str): Name of the second muscle.
        """
        self.m_muscleLabel1 = label1
        self.m_muscleLabel2 = label2

    def setContext(self,context:str):
        """
        Set the context for the coactivation analysis.

        Args:
            context (str): The context of the analysis (e.g., 'Left', 'Right').
        """
        self.m_context = context

    def __setLayer(self):
        """
        Internal method to set up the plot layers.

        Configures the plot layout, titles, and axes for EMG coactivation visualization.
        """

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=" Coactivation : " + self.m_muscleLabel1 + " Vs " + self.m_muscleLabel2
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(1,1,1)


        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_ylabel("Emg Unit",size=8)


    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Set a normative dataset for comparison.

        Note: Currently, displaying normal EMG data is not implemented.

        Args:
            iNormativeDataSet (NormativeData): Instance of NormativeData for comparison.
        """
        LOGGER.logger.warning("[pyCGM2] - pycgm2 not include and display normal emg data")
        pass

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Set a concrete plot function for the viewer.

        Args:
            concreteplotFunction (Callable): A specific plot function from pyCGM2.Report.plot.
        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        """
        Internal method to set the data for the coactivation plot.

        Processes and plots the EMG data for muscle coactivation analysis.
        """

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
        """
        Plot the coactivation panel with EMG data.

        Generates and returns the final plot figure with configured EMG coactivation data.
        """

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig

class EnvEmgGaitPlotPanelViewer(plotViewers.PlotViewer):
    """
    A viewer for plotting EMG envelops during gait analysis.

    This viewer creates plots of EMG envelops for selected channels, providing insights 
    into muscle activity throughout gait cycles.

    Args:
        iAnalysis (Analysis): An Analysis instance containing processed EMG data.
        pointLabelSuffix (str, optional): A suffix added to EMG outputs for custom labeling.
    """

    def __init__(self,iAnalysis:Analysis,pointLabelSuffix:Optional[str]=None):
        """Initialize the EnvEmgGaitPlotPanelViewer."""

        super(EnvEmgGaitPlotPanelViewer, self).__init__(iAnalysis)

        self.emgs = []
        self.m_normalActivEmgs = []

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

        self.m_normalizedEmgFlag = False
        self.m_selectChannels = None

    def selectEmgChannels(self,channelNames:List[str]):
        """
        Select specific EMG channels for plotting.

        Args:
            channelNames (List[str]): List of EMG channel names to be plotted.
        """
        self.m_selectChannels = channelNames

    def setEmgManager(self,emgManager:EmgManager):
        """
        Set the EMG Manager for the viewer.

        Args:
            emgManager (EmgManager): An instance of EmgManager containing EMG processing details.
        """
        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()


    def setNormalizedEmgFlag(self,flag:bool):
        """Enable/Disable amplitude-normalized emg

        Args:
            flag (bool): boolean flag

        """
        self.m_normalizedEmgFlag = flag


    def __setLayer(self):
        """
        Internal method to set up the plot layers.

        Configures the layout, titles, and axes for EMG envelop visualization during gait.
        """

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



    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Set a normative dataset for comparison (not implemented).

        Args:
            iNormativeDataSet (NormativeData): Instance of NormativeData for comparison.
        """
        LOGGER.logger.warning("[pyCGM2] - pycgm2 not include and display normal emg data")
        pass

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Set a specific plot function for the viewer.

        Args:
            concreteplotFunction (Callable): A plot function from pyCGM2.Report.plot.
        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        """
        Internal method to set the data for the EMG gait plot.

        Processes and plots the EMG data for gait cycle analysis.
        """

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
        """
        Plot the EMG gait panel.

        Generates and returns the final plot figure with configured EMG data for gait analysis.
        """

        self.__setLayer()
        self.__setData()

        return self.fig



class MultipleAnalysis_EnvEmgPlotPanelViewer(plotViewers.PlotViewer):
    """
    A viewer for plotting EMG envelops from multiple Analysis instances.

    This viewer is used to compare EMG envelops across different Analysis instances, 
    which can be useful for comparing muscle activities under different conditions.

    Args:
        iAnalyses (List[Analysis]): A list of Analysis instances to compare.
        legends (List[str]): Labels associated with each Analysis instance for legend display.
        pointLabelSuffix (str, optional): A suffix added to EMG outputs for custom labeling.
    """

    def __init__(self,iAnalyses:List[Analysis],legends:List[str],pointLabelSuffix:Optional[str]=None):
        """Initialize the MultipleAnalysis_EnvEmgPlotPanelViewer."""

        super(MultipleAnalysis_EnvEmgPlotPanelViewer, self).__init__(iAnalyses)


        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")


        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input


        self.emgs = []
        self.m_normalActivEmgs = []


        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normalizedEmgFlag = False
        self.m_legends = legends
        self.m_selectChannels = None

    def selectEmgChannels(self,channelNames:List[str]):
        """
        Select specific EMG channels for visualization across multiple analyses.

        Args:
            channelNames (List[str]): List of EMG channel names to include in the plots.
        """
        self.m_selectChannels = channelNames

    def setEmgManager(self,emgManager:EmgManager):
        """
        Set the EMG Manager for the viewer.

        Args:
            emgManager (EmgManager): An instance of EmgManager containing EMG processing details.
        """
        self.m_emgmanager = emgManager
        if self.m_selectChannels is None:
            self.m_selectChannels = self.m_emgmanager.getChannels()

    def setNormalizedEmgFlag(self,flag:bool):
        """Enable/Disable amplitude-normalized emg

        Args:
            flag (bool): boolean flag

        """
        self.m_normalizedEmgFlag = flag


    def __setLayer(self):
        """
        Internal method to set up the plot layers for multiple analyses.

        Configures the layout, titles, and axes for EMG envelop visualization.
        """

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
        """
        Internal method to set the legend for the plot.

        Args:
            axisIndex (int): Index of the axis where the legend will be placed.
        """
        self.fig.axes[axisIndex].legend(fontsize=6)


    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Set a normative dataset for comparison.

        Note: Currently, displaying normal EMG data is not implemented.

        Args:
            iNormativeDataSet (NormativeData): Instance of NormativeData for comparison.
        """
        LOGGER.logger.warning("[pyCGM2] - pycgm2 not include and display normal emg data")
        pass

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Set a specific plot function for the viewer.

        Args:
            concreteplotFunction (Callable): A plot function from pyCGM2.Report.plot.
        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setData(self):
        """
        Internal method to set the data for the EMG plot panel from multiple analyses.

        Processes and plots EMG data from different analyses for comparative visualization.
        """
        #suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

        colormap_i_left=[plt.cm.Reds(k) for k in np.linspace(0.2, 1, len(self.m_analysis))]
        colormap_i_left = [(0,0,0)] + colormap_i_left
        colormap_i_right=[plt.cm.Blues(k) for k in np.linspace(0.2, 1, len(self.m_analysis))]
        colormap_i_right = [(0,0,0)] + colormap_i_right


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
        """
        Plot the EMG panel with data from multiple analyses.

        Generates and returns the final plot figure with configured EMG data for comparative analysis.
        """

        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

        # normative dataset not implemented

        return self.fig
