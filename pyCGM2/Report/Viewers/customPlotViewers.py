"""
Module contains `plotViewers` to display custom panel
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



class SaggitalGagePlotViewer(plotViewers.PlotViewer):
    """
    A viewer for creating plots that replicate the sagittal gait analysis approach as proposed in 
    'The Identification and Treatment of Gait Problems in Cerebral Palsy' by James R. Gage MD, et al.

    This class is designed to produce plots that integrate kinematic, kinetic, and electromyographic data 
    in a manner consistent with the methodologies described in the referenced work. It provides a 
    comprehensive view of gait analysis by combining these different data types.

    Args:
        iAnalysis (Analysis): An Analysis instance containing the gait analysis data.
        emgManager (EmgManager): An EmgManager instance managing the electromyographic data.
        emgType (str): type of emg signal to plot. Defaults to `Envelope`, choice: `Raw` or `Rectify`.
        pointLabelSuffix (Optional[str]): An optional suffix for the point labels. Defaults to None.
    """

    def __init__(self,iAnalysis:Analysis,emgManager:EmgManager,emgType:str="Envelope", pointLabelSuffix:Optional[str]=None):
        super(SaggitalGagePlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_emgType = emgType

        self.m_emgManager = emgManager

        self.m_concretePlotFunction = None

    def setConcretePlotFunction(self, concreteplotFunction):
        pass

    def __setLayer(self):
        """internal method to set the plot layer
        """

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        self.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        self.fig.suptitle("Saggital Plots")
        widths = [1, 1, 1]
        heights = [1, 0.5, 0.5, 1, 0.5, 0.5, 1]
        grid = self.fig.add_gridspec(ncols=3, nrows=7, width_ratios=widths,
                                  height_ratios=heights)


        # fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
        ax0 = self.fig.add_subplot(grid[0,0]); ax0.set_title("Hip Flex" ,size=8)# Hip Flex
        ax1 = self.fig.add_subplot(grid[0,1]); ax1.set_title("Knee Flex" ,size=8)# kne Ext
        ax2 = self.fig.add_subplot(grid[0,2]); ax2.set_title("Ankle Flex" ,size=8)# ankle

        ax3 = self.fig.add_subplot(grid[1,0]); ax3.set_title("SEMITE" ,size=8)# ST
        ax4 = self.fig.add_subplot(grid[1,1]); ax4.set_title("RECFEM" ,size=8)# RF
        ax5 = self.fig.add_subplot(grid[1,2]); ax5.set_title("SOLEUS" ,size=8)# SOL

        ax6 = self.fig.add_subplot(grid[2,0])# NONE
        ax7 = self.fig.add_subplot(grid[2,1]); ax7.set_title("VASLAT" ,size=8)# VL
        ax8 = self.fig.add_subplot(grid[2,2])# NONE

        ax9 = self.fig.add_subplot(grid[3,0]); ax9.set_title("Hip Ext Moment" ,size=8)# hip Moment
        ax10 = self.fig.add_subplot(grid[3,1]); ax10.set_title("Knee Ext Moment" ,size=8)# knee
        ax11 = self.fig.add_subplot(grid[3,2]); ax11.set_title("Ankle Ext Moment" ,size=8)# ankle

        ax12 = self.fig.add_subplot(grid[4,0]); ax12.set_title("RECFEM" ,size=8)# RF
        ax13 = self.fig.add_subplot(grid[4,1]); ax13.set_title("SEMITE" ,size=8)# HAM
        ax14 = self.fig.add_subplot(grid[4,2]); ax14.set_title("TIBANT" ,size=8)# TA

        ax15 = self.fig.add_subplot(grid[5,0])
        ax16 = self.fig.add_subplot(grid[5,1]); ax16.set_title("SOLEUS" ,size=8)# SOL
        ax17 = self.fig.add_subplot(grid[5,2])

        ax18 = self.fig.add_subplot(grid[6,0]); ax18.set_title("Hip Power" ,size=8)# hip Power
        ax19 = self.fig.add_subplot(grid[6,1]); ax19.set_title("Knee Power" ,size=8)# knee
        ax20 = self.fig.add_subplot(grid[6,2]); ax20.set_title("Ankle Power" ,size=8)# ankle


        ax6.set_visible(False)
        ax8.set_visible(False)
        ax15.set_visible(False)
        ax17.set_visible(False)

        for axIt in [ax3,ax4,ax5,ax7,ax12,ax13,ax14,ax16]:
            axIt.get_yaxis().set_visible(False)


        ax18.set_xlabel("Cycle %",size=8)
        ax19.set_xlabel("Cycle %",size=8)
        ax20.set_xlabel("Cycle %",size=8)

        if not self.m_automaticYlim_flag:
                ax0.set_ylim( [-20,70])
                ax1.set_ylim([-15,75])
                ax2.set_ylim([-30,30])
               
                ax9.set_ylim([-2.0 *1000.0, 3.0*1000.0])
                ax10.set_ylim([-1.0*1000.0, 1.0*1000.0])
                ax11.set_ylim([-1.0*1000.0, 3.0*1000.0])

                ax18.set_ylim(  [-3.0, 3.0])
                ax19.set_ylim(  [-3.0, 3.0])
                ax20.set_ylim([-2.0,5.0])

    def setNormativeDataset(self,iNormativeDataSet):
        """
        Set a normative dataset

        Args:
            iNormativeDataSet (NormativeData): Instance of NormativeData for comparison.
        """

        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        """
        Internal method to set the data 
        """
        footOff_L = self.m_analysis.emgStats.pst['stancePhase', "Left"]["mean"]
        footOff_R = self.m_analysis.emgStats.pst['stancePhase', "Left"]["mean"]

        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""
        if self.m_emgType=="Raw":
            emgEnvSuff = ""
        elif self.m_emgType=="Rectify":
            emgEnvSuff = "_Rectify"
        else:
            emgEnvSuff = "_Rectify_Env"

        
        plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                "LHipAngles"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                "LKneeAngles"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                "LAnkleAngles"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[0],self.m_analysis.kinematicStats,
                "RHipAngles"+suffixPlus,"Right",0, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[1],self.m_analysis.kinematicStats,
                "RKneeAngles"+suffixPlus,"Right",0, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[2],self.m_analysis.kinematicStats,
                "RAnkleAngles"+suffixPlus,"Right",0, color="blue", customLimits=None)

        
        
        plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.emgStats,
                          self.m_emgManager.getChannel("SEMITE","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[3],self.m_analysis.emgStats,
                          self.m_emgManager.getChannel("SEMITE","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[3],"SEMITE", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[3],"SEMITE", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)
        
        

        plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("RECFEM","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[4],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("RECFEM","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[4],"RECFEM", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[4],"RECFEM", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)
        # self.fig.axes[4].plot(self.m_analysis.emgStats.data[self.m_emgMetadata.getChannel("RF","Left"),"Left"]["values"][0])

        plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SOLEUS","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[5],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SOLEUS","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[5],"SOLEUS", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[5],"SOLEUS", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)
        # self.fig.axes[5].plot(self.m_analysis.emgStats.data[self.m_emgMetadata.getChannel("SOL","Left"),"Left"]["values"][0])

        plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("VASLAT","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[7],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("VASLAT","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[6],"VASTLAT", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[6],"VASTLAT", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)

        plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kineticStats,
                "LHipMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kineticStats,
                "LKneeMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kineticStats,
                "LAnkleMoment"+suffixPlus,"Left",0, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[9],self.m_analysis.kineticStats,
                "RHipMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[10],self.m_analysis.kineticStats,
                "RKneeMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[11],self.m_analysis.kineticStats,
                "RAnkleMoment"+suffixPlus,"Right",0, color="blue", customLimits=None)
        
        

        plot.gaitMeanPlot(self.fig.axes[12],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("RECFEM","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[12],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("RECFEM","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[12],"RECFEM", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[12],"RECFEM", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)

        plot.gaitMeanPlot(self.fig.axes[13],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SEMITE","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[13],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SEMITE","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[13],"SEMITE", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[13],"SEMITE", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)

        plot.gaitMeanPlot(self.fig.axes[14],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("TIBANT","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[14],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("TIBANT","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[14],"TIBANT", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[14],"TIBANT", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)

        plot.gaitMeanPlot(self.fig.axes[16],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SOLEUS","Left")+emgEnvSuff,"Left",0, color="red",alpha=0.5)
        plot.gaitMeanPlot(self.fig.axes[16],self.m_analysis.emgStats,
                self.m_emgManager.getChannel("SOLEUS","Right")+emgEnvSuff,"Right",0, color="blue",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[16],"SOLEUS", footOff_L,position="Upper",edgecolor="red",alpha=0.5)
        plot.addNormalActivationLayer(self.fig.axes[16],"SOLEUS", footOff_R,position="Lower",edgecolor="blue",alpha=0.5)


        plot.gaitDescriptivePlot(self.fig.axes[18],self.m_analysis.kineticStats,
                "LHipPower"+suffixPlus,"Left",2, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[19],self.m_analysis.kineticStats,
                "LKneePower"+suffixPlus,"Left",2, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[20],self.m_analysis.kineticStats,
                "LAnklePower"+suffixPlus,"Left",2, color="red", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[18],self.m_analysis.kineticStats,
                "RHipPower"+suffixPlus,"Right",2, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[19],self.m_analysis.kineticStats,
                "RKneePower"+suffixPlus,"Right",2, color="blue", customLimits=None)
        plot.gaitDescriptivePlot(self.fig.axes[20],self.m_analysis.kineticStats,
                "RAnklePower"+suffixPlus,"Right",2, color="blue", customLimits=None)

#
    def plotPanel(self):
        """
        Plot the Gage-inspired Sagittal Gait panel.

        Generates and returns the final plot figure 
        """

        # if self.m_concretePlotFunction is None:
        #     raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["HipAngles"]["mean"].shape[0]),
                self.m_normativeData["HipAngles"]["mean"][:,0]-self.m_normativeData["HipAngles"]["sd"][:,0],
                self.m_normativeData["HipAngles"]["mean"][:,0]+self.m_normativeData["HipAngles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["KneeAngles"]["mean"].shape[0]),
                self.m_normativeData["KneeAngles"]["mean"][:,0]-self.m_normativeData["KneeAngles"]["sd"][:,0],
                self.m_normativeData["KneeAngles"]["mean"][:,0]+self.m_normativeData["KneeAngles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[2].fill_between(np.linspace(0,100,self.m_normativeData["AnkleAngles"]["mean"].shape[0]),
                self.m_normativeData["AnkleAngles"]["mean"][:,0]-self.m_normativeData["AnkleAngles"]["sd"][:,0],
                self.m_normativeData["AnkleAngles"]["mean"][:,0]+self.m_normativeData["AnkleAngles"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)



            self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
                self.m_normativeData["HipMoment"]["mean"][:,0]-self.m_normativeData["HipMoment"]["sd"][:,0],
                self.m_normativeData["HipMoment"]["mean"][:,0]+self.m_normativeData["HipMoment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[10].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
                self.m_normativeData["KneeMoment"]["mean"][:,0]-self.m_normativeData["KneeMoment"]["sd"][:,0],
                self.m_normativeData["KneeMoment"]["mean"][:,0]+self.m_normativeData["KneeMoment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
                self.m_normativeData["AnkleMoment"]["mean"][:,0]-self.m_normativeData["AnkleMoment"]["sd"][:,0],
                self.m_normativeData["AnkleMoment"]["mean"][:,0]+self.m_normativeData["AnkleMoment"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)



            self.fig.axes[18].fill_between(np.linspace(0,100,self.m_normativeData["HipPower"]["mean"].shape[0]),
                self.m_normativeData["HipPower"]["mean"][:,0]-self.m_normativeData["HipPower"]["sd"][:,0],
                self.m_normativeData["HipPower"]["mean"][:,0]+self.m_normativeData["HipPower"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[19].fill_between(np.linspace(0,100,self.m_normativeData["KneePower"]["mean"].shape[0]),
                self.m_normativeData["KneePower"]["mean"][:,0]-self.m_normativeData["KneePower"]["sd"][:,0],
                self.m_normativeData["KneePower"]["mean"][:,0]+self.m_normativeData["KneePower"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

            self.fig.axes[20].fill_between(np.linspace(0,100,self.m_normativeData["AnklePower"]["mean"].shape[0]),
                self.m_normativeData["AnklePower"]["mean"][:,0]-self.m_normativeData["AnklePower"]["sd"][:,0],
                self.m_normativeData["AnklePower"]["mean"][:,0]+self.m_normativeData["AnklePower"]["sd"][:,0],
                facecolor="green", alpha=0.5,linewidth=0)

        return self.fig