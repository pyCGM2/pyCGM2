import numpy as np
import matplotlib.pyplot as plt
import copy

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Report.Viewers import plotViewers


from pyCGM2.Report.normativeDatasets import NormativeData
from pyCGM2.Processing.analysis import Analysis

from typing import Optional, Callable

class MuscleNormalizedPlotPanelViewer(plotViewers.PlotViewer):
    """plot emg envelops

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        pointLabelSuffix (str,Optional[None]): suffix added to emg outputs


    """

    def __init__(self,iAnalysis:Analysis,pointLabelSuffix:Optional[str]=None):

        super(MuscleNormalizedPlotPanelViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

        self.m_normalisationSuffix = ""

        self.m_normativeData=None


    def setNormalizationSuffix(self,suffix:str):
        self.m_normalisationSuffix = "_"+suffix

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Time-normalized Muscle Plot \n """ 
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

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

            ax.set_xlabel("Cycle %",size=8)
            if self.m_muscleOutputType == "MuscleLength":
                ax.set_ylabel("Length (m)",size=8)   

        i=0
        for muscle in self.m_muscles:
            self.fig.axes[i].set_title(muscle,size=8)
            
            i+=1

        if not self.m_automaticYlim_flag:
            for axisIt in self.fig.axes:
                axisIt.set_ylim([0.5,1.5])

    def __setData(self):

        muscles_leftContext =[]
        muscles_rightContext =[]

        if self.m_muscleOutputType == "MuscleLength":

            

            for keys in self.m_analysis.muscleGeometryStats.data:
                
                muscle = keys[0]
                if keys[1] == "Left": muscles_leftContext.append(muscle)
                if keys[1] == "Right": muscles_rightContext.append(muscle)
            

            i=0
            for muscle in self.m_muscles:
                leftLabel = muscle+"_l"+"["+self.m_muscleOutputType+"]"+self.m_normalisationSuffix
                rightLabel = muscle+"_r"+"["+self.m_muscleOutputType+"]"+self.m_normalisationSuffix

                if leftLabel in muscles_leftContext:
                    self.m_concretePlotFunction(self.fig.axes[i],self.m_analysis.muscleGeometryStats,
                                                leftLabel,"Left",0,
                                                color="red",
                                                customLimits=None)

                if rightLabel in muscles_rightContext:
                    self.m_concretePlotFunction(self.fig.axes[i],self.m_analysis.muscleGeometryStats,
                                                rightLabel,"Right",0,
                                                color="blue",
                                                customLimits=None)

                i+=1
        else:
            LOGGER.logger.error("[pyCGM2] -  muscle panel other than muscle length not implemented yet")
                

    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """ Set the normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): a normative dataset instance

        """
        self.m_normativeData = iNormativeDataSet.data

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """set a concrete plot function

        Args:
            concreteplotFunction (Callable): a plot function of pyCGM2.Report.plot

        """
        self.m_concretePlotFunction = concreteplotFunction

    def setMuscles(self,iMuscles:list):
        self.m_muscles= iMuscles

    def setMuscleOutputType(self,type:str):
        self.m_muscleOutputType= type


    def plotPanel(self):
        """plot the panel"""

        self.__setLayer()
        self.__setData()


        if self.m_normativeData is not None:
            i=0
            for muscle in self.m_muscles:
                label = muscle+"["+self.m_muscleOutputType+"]"+self.m_normalisationSuffix

                if label in self.m_normativeData:
                    self.fig.axes[i].fill_between(np.linspace(0,100,self.m_normativeData[label]["mean"].shape[0]),
                        self.m_normativeData[label]["mean"][:,0]-self.m_normativeData[label]["sd"][:,0],
                        self.m_normativeData[label]["mean"][:,0]+self.m_normativeData[label]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)
                else:
                    LOGGER.logger.warning("[pycgm2] label (%s) not detected in the normative dataset"%(label))

                i+=1

        return self.fig