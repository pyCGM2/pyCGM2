# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

"""
Module contains `plotViewers` for displaying convenient gait plot
"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

import matplotlib.pyplot as plt

# pyCGM2
import pyCGM2
from pyCGM2 import enums
from pyCGM2.Report import plot



class AbstractPlotViewer(object):
    def __init__(self,input,AutomaticYlimits=False):
        self.m_input =input
        self.m_automaticYlim_flag = AutomaticYlimits

    def setNormativeData(self):
        pass

    def __setLayer(self):
        pass

    def __setData(self):
        pass

    def plotPanel(self):
        pass

    def setAutomaticYlimits(self,bool):
        self.m_automaticYlim_flag = bool

class SpatioTemporalPlotViewer(AbstractPlotViewer):
    """ Plot viewer to display spatio-temporal parameters as histogram

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance

    """

    def __init__(self,iAnalysis):

        super(SpatioTemporalPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

    def setNormativeDataset(self,iNormativeDataSet):
        """Set the normative dataset


        Args:
             iNormativeDataSet(pyCGM2.Report.normativeDatasets.NormalSTP): a spatio-temporal normative dataset instance
        """

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
                                title="cadence (strides/min)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[4],self.m_analysis.stpStats,
                                "stepLength",
                                overall= False,
                                title="Step length (m)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[5],self.m_analysis.stpStats,
                                "stepDuration",
                                overall= False,
                                title="Step time (s)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[6],self.m_analysis.stpStats,
                                "stancePhase",
                                overall= False,
                                title="Stance time (% of gait cycle)", xlabel=None,xlim=None)

        plot.stpHorizontalHistogram(self.fig.axes[7],self.m_analysis.stpStats,
                                "doubleStance1",
                                overall= False,
                                title="Initial double limb support (% of gait cycle)", xlabel=None,xlim=None)


    def plotPanel(self):
        "Plot the panel"
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

        return self.fig

class GpsMapPlotViewer(AbstractPlotViewer):
    """ Plot viewer to display GPS and MAP panel

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance

    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):



        super(GpsMapPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix

    # def setNormativeDataset(self,iNormativeDataSet):
    #     pass

    def __setLayer(self):
        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u"""Mouvement  Analysis Profile \n """
        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(1,1,1)


    def __setData(self):

        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""
        N = 9
        overall = (self.m_analysis.gvs["LPelvisAngles"+suffixPlus,"Left"]["mean"][0],
                     0,
                     0,
                     self.m_analysis.gvs["LPelvisAngles"+suffixPlus,"Left"]["mean"][1],
                     0,
                     self.m_analysis.gvs["LPelvisAngles"+suffixPlus,"Left"]["mean"][2],
                     0,
                     0,
                      self.m_analysis.gps["Overall"]["mean"][0])
        left = (0,
                 self.m_analysis.gvs["LHipAngles"+suffixPlus,"Left"]["mean"][0],
                 self.m_analysis.gvs["LKneeAngles"+suffixPlus,"Left"]["mean"][0],
                  0,
                  self.m_analysis.gvs["LHipAngles"+suffixPlus,"Left"]["mean"][1],
                  0,
                  self.m_analysis.gvs["LHipAngles"+suffixPlus,"Left"]["mean"][2],
                  self.m_analysis.gvs["LFootProgressAngles"+suffixPlus,"Left"]["mean"][2],
                  self.m_analysis.gps["Context"]["Left"]["mean"][0])

        right = (0,
                 self.m_analysis.gvs["RHipAngles"+suffixPlus,"Right"]["mean"][0],
                 self.m_analysis.gvs["RKneeAngles"+suffixPlus,"Right"]["mean"][0],
                  0,
                  self.m_analysis.gvs["RHipAngles"+suffixPlus,"Right"]["mean"][1],
                  0,
                  self.m_analysis.gvs["RHipAngles"+suffixPlus,"Right"]["mean"][2],
                  self.m_analysis.gvs["RFootProgressAngles"+suffixPlus,"Right"]["mean"][2],
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
        """ plat the panel"""
        self.__setLayer()
        self.__setData()

        return self.fig

class NormalizedKinematicsPlotViewer(AbstractPlotViewer):
    """ Plot time-Normalized Kinematics

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis instance`
        pointLabelSuffix (str): suffix added model outputs
        bodyPart (enums.BodyPartPlot,Optional[enums.BodyPartPlot.LowerLimb]): body part

    """

    def __init__(self,iAnalysis,pointLabelSuffix=None,bodyPart=enums.BodyPartPlot.LowerLimb):

        super(NormalizedKinematicsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_bodyPart = bodyPart

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_concretePlotFunction = None


    def setConcretePlotFunction(self, concreteplotFunction):
        """set a plot function ( see `plot`)

        Args:
            concreteplotFunction (function): plot function

        """
        self.m_concretePlotFunction = concreteplotFunction


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")

        if self.m_concretePlotFunction.__name__ in ["descriptivePlot","gaitDescriptivePlot"]:
            title=u""" Descriptive Time-normalized Kinematics \n """
        elif self.m_concretePlotFunction.__name__ in ["consistencyPlot","gaitConsistencyPlot"]:
            title=u""" Consistency Time-normalized Kinematics \n """
        else :
            title=u"""\n"""

        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax0 = plt.subplot(6,3,1)
        ax1 = plt.subplot(6,3,2)
        ax2 = plt.subplot(6,3,3)
        ax3 = plt.subplot(6,3,4)
        ax4 = plt.subplot(6,3,5)
        ax5 = plt.subplot(6,3,6)
        ax6 = plt.subplot(6,3,7)
        ax7 = plt.subplot(6,3,8)
        ax8 = plt.subplot(6,3,9)
        ax9 = plt.subplot(6,3,10)
        ax10 = plt.subplot(6,3,11)
        ax11 = plt.subplot(6,3,12)
        ax12 = plt.subplot(6,3,13)
        ax13 = plt.subplot(6,3,14)
        ax14 = plt.subplot(6,3,15)

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

        ax12.set_xlabel("Cycle %",size=8)
        ax13.set_xlabel("Cycle %",size=8)
        ax14.set_xlabel("Cycle %",size=8)


        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:

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

            if not self.m_automaticYlim_flag:
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

        elif self.m_bodyPart == enums.BodyPartPlot.Trunk:

            ax0.set_title("Pelvis Tilt" ,size=8)
            ax1.set_title("Pelvis Obliquity" ,size=8)
            ax2.set_title("Pelvis Rotation" ,size=8)
            ax3.set_title("Spine Flexion" ,size=8)
            ax4.set_title("Spine Adduction" ,size=8)
            ax5.set_title("Spine Rotation" ,size=8)
            ax6.set_title("Thorax Anteversion" ,size=8)
            ax7.set_title("Thorax Obliquity" ,size=8)
            ax8.set_title("Thorax Rotation" ,size=8)

            ax9.set_title("Neck Forward Tilt" ,size=8)
            ax10.set_title("Neck Lateral Tilt" ,size=8)
            ax11.set_title("Neck Rotation" ,size=8)

            ax12.set_title("Head Forward Tilt" ,size=8)
            ax13.set_title("Head Lateral Tilt" ,size=8)
            ax14.set_title("Head Rotation" ,size=8)

            if not self.m_automaticYlim_flag:
                ax0.set_ylim([0,60])
                ax1.set_ylim([-30,30])
                ax2.set_ylim([-30,30])

                ax3.set_ylim( [-30,30])
                ax4.set_ylim([-30,30])
                ax5.set_ylim([-30,30])

                ax6.set_ylim([-30,30])
                ax7.set_ylim([-30,30])
                ax8.set_ylim([-30,30])

                ax9.set_ylim([-30,30])
                ax10.set_ylim([-30,30])
                ax11.set_ylim([-30,30])

                ax12.set_ylim([-50,30])
                ax13.set_ylim([-30,30])
                ax14.set_ylim([-30,30])

        elif self.m_bodyPart == enums.BodyPartPlot.UpperLimb:

            ax0.set_title("Shoulder flexion" ,size=8)
            ax1.set_title("Shoulder Adduction" ,size=8)
            ax2.set_title("Shoulder Rotation" ,size=8)
            ax3.set_title("Elbow Flexion" ,size=8)
            # ax4.set_title("Elbow Adduction" ,size=8)
            # ax5.set_title("Spine Rotation" ,size=8)
            ax6.set_title("Ulnar Deviation" ,size=8)
            ax7.set_title("Wrist Extension" ,size=8)
            ax8.set_title("Wrist Rotation" ,size=8)

            if not self.m_automaticYlim_flag:
                ax0.set_ylim([-60,60])
                ax1.set_ylim([-30,30])
                ax2.set_ylim([-30,30])

                ax3.set_ylim( [-60,60])
                ax4.set_ylim([-30,30])
                ax5.set_ylim([-30,30])

                ax6.set_ylim([-30,30])
                ax7.set_ylim([-30,30])
                ax8.set_ylim([0,160])

                ax9.set_ylim([-30,30])
                ax10.set_ylim([-30,30])
                ax11.set_ylim([-30,30])

                ax12.set_ylim([-50,30])
                ax13.set_ylim([-30,30])
                ax14.set_ylim([-30,30])

    def setNormativeDataset(self,iNormativeDataSet):
        """ Set the normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): a normative dataset instance

        """
        # iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:

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

        elif self.m_bodyPart == enums.BodyPartPlot.Trunk:
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

            # Spine
            self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "LSpineAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "RSpineAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "LSpineAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
                    "RSpineAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "LSpineAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
                    "RSpineAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

            # Thorax
            self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "LThoraxAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "RThoraxAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "LThoraxAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "RThoraxAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "LThoraxAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "RThoraxAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)


            # Neck
            self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "LNeckAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[9],self.m_analysis.kinematicStats,
                    "RNeckAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "LNeckAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[10],self.m_analysis.kinematicStats,
                    "RNeckAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "LNeckAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[11],self.m_analysis.kinematicStats,
                    "RNeckAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)


            # Head
            self.m_concretePlotFunction(self.fig.axes[12],self.m_analysis.kinematicStats,
                    "LHeadAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[12],self.m_analysis.kinematicStats,
                    "RHeadAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[13],self.m_analysis.kinematicStats,
                    "LHeadAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[13],self.m_analysis.kinematicStats,
                    "RHeadAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[14],self.m_analysis.kinematicStats,
                    "LHeadAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[14],self.m_analysis.kinematicStats,
                    "RHeadAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)



        elif self.m_bodyPart == enums.BodyPartPlot.UpperLimb:
            # shoulder
            self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "LShoulderAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kinematicStats,
                    "RShoulderAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "LShoulderAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kinematicStats,
                    "RShoulderAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "LShoulderAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kinematicStats,
                    "RShoulderAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

            # elbow
            self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "LElbowAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[3],self.m_analysis.kinematicStats,
                    "RElbowAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            # self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
            #         "LElbowAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            # self.m_concretePlotFunction(self.fig.axes[4],self.m_analysis.kinematicStats,
            #         "LElbowAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)
            #
            # self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
            #         "LElbowAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            # self.m_concretePlotFunction(self.fig.axes[5],self.m_analysis.kinematicStats,
            #         "RElbowAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)

            # wrist
            self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "LWristAngles"+suffixPlus,"Left",0, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[6],self.m_analysis.kinematicStats,
                    "RWristAngles"+suffixPlus,"Right",0, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "LWristAngles"+suffixPlus,"Left",1, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[7],self.m_analysis.kinematicStats,
                    "RWristAngles"+suffixPlus,"Right",1, color="blue",customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "LWristAngles"+suffixPlus,"Left",2, color="red",customLimits=None)
            self.m_concretePlotFunction(self.fig.axes[8],self.m_analysis.kinematicStats,
                    "RWristAngles"+suffixPlus,"Right",2, color="blue",customLimits=None)



    def plotPanel(self):
        """ Plot the panel"""

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:

                if "PelvisAngles" in self.m_normativeData.keys():
                    self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,0]-self.m_normativeData["PelvisAngles"]["sd"][:,0],
                        self.m_normativeData["PelvisAngles"]["mean"][:,0]+self.m_normativeData["PelvisAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,1]-self.m_normativeData["PelvisAngles"]["sd"][:,1],
                        self.m_normativeData["PelvisAngles"]["mean"][:,1]+self.m_normativeData["PelvisAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[2].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,2]-self.m_normativeData["PelvisAngles"]["sd"][:,2],
                        self.m_normativeData["PelvisAngles"]["mean"][:,2]+self.m_normativeData["PelvisAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "HipAngles" in self.m_normativeData.keys():
                    self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["HipAngles"]["mean"].shape[0]),
                        self.m_normativeData["HipAngles"]["mean"][:,0]-self.m_normativeData["HipAngles"]["sd"][:,0],
                        self.m_normativeData["HipAngles"]["mean"][:,0]+self.m_normativeData["HipAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["HipAngles"]["mean"].shape[0]),
                        self.m_normativeData["HipAngles"]["mean"][:,1]-self.m_normativeData["HipAngles"]["sd"][:,1],
                        self.m_normativeData["HipAngles"]["mean"][:,1]+self.m_normativeData["HipAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["HipAngles"]["mean"].shape[0]),
                        self.m_normativeData["HipAngles"]["mean"][:,2]-self.m_normativeData["HipAngles"]["sd"][:,2],
                        self.m_normativeData["HipAngles"]["mean"][:,2]+self.m_normativeData["HipAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "KneeAngles" in self.m_normativeData.keys():
                    self.fig.axes[6].fill_between(np.linspace(0,100,self.m_normativeData["KneeAngles"]["mean"].shape[0]),
                        self.m_normativeData["KneeAngles"]["mean"][:,0]-self.m_normativeData["KneeAngles"]["sd"][:,0],
                        self.m_normativeData["KneeAngles"]["mean"][:,0]+self.m_normativeData["KneeAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["KneeAngles"]["mean"].shape[0]),
                        self.m_normativeData["KneeAngles"]["mean"][:,1]-self.m_normativeData["KneeAngles"]["sd"][:,1],
                        self.m_normativeData["KneeAngles"]["mean"][:,1]+self.m_normativeData["KneeAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["KneeAngles"]["mean"].shape[0]),
                        self.m_normativeData["KneeAngles"]["mean"][:,2]-self.m_normativeData["KneeAngles"]["sd"][:,2],
                        self.m_normativeData["KneeAngles"]["mean"][:,2]+self.m_normativeData["KneeAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "AnkleAngles" in self.m_normativeData.keys():
                    self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["AnkleAngles"]["mean"].shape[0]),
                        self.m_normativeData["AnkleAngles"]["mean"][:,0]-self.m_normativeData["AnkleAngles"]["sd"][:,0],
                        self.m_normativeData["AnkleAngles"]["mean"][:,0]+self.m_normativeData["AnkleAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[10].fill_between(np.linspace(0,100,self.m_normativeData["AnkleAngles"]["mean"].shape[0]),
                        self.m_normativeData["AnkleAngles"]["mean"][:,1]-self.m_normativeData["AnkleAngles"]["sd"][:,1],
                        self.m_normativeData["AnkleAngles"]["mean"][:,1]+self.m_normativeData["AnkleAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "FootProgressAngles" in self.m_normativeData.keys():
                    self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["FootProgressAngles"]["mean"].shape[0]),
                        self.m_normativeData["FootProgressAngles"]["mean"][:,2]-self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                        self.m_normativeData["FootProgressAngles"]["mean"][:,2]+self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

            elif self.m_bodyPart == enums.BodyPartPlot.Trunk:


                if "PelvisAngles" in self.m_normativeData.keys():
                    self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,0]-self.m_normativeData["PelvisAngles"]["sd"][:,0],
                        self.m_normativeData["PelvisAngles"]["mean"][:,0]+self.m_normativeData["PelvisAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,1]-self.m_normativeData["PelvisAngles"]["sd"][:,1],
                        self.m_normativeData["PelvisAngles"]["mean"][:,1]+self.m_normativeData["PelvisAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[2].fill_between(np.linspace(0,100,self.m_normativeData["PelvisAngles"]["mean"].shape[0]),
                        self.m_normativeData["PelvisAngles"]["mean"][:,2]-self.m_normativeData["PelvisAngles"]["sd"][:,2],
                        self.m_normativeData["PelvisAngles"]["mean"][:,2]+self.m_normativeData["PelvisAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "SpineAngles" in self.m_normativeData.keys():
                    self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["SpineAngles"]["mean"].shape[0]),
                        self.m_normativeData["SpineAngles"]["mean"][:,0]-self.m_normativeData["SpineAngles"]["sd"][:,0],
                        self.m_normativeData["SpineAngles"]["mean"][:,0]+self.m_normativeData["SpineAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["SpineAngles"]["mean"].shape[0]),
                        self.m_normativeData["SpineAngles"]["mean"][:,1]-self.m_normativeData["SpineAngles"]["sd"][:,1],
                        self.m_normativeData["SpineAngles"]["mean"][:,1]+self.m_normativeData["SpineAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["SpineAngles"]["mean"].shape[0]),
                        self.m_normativeData["SpineAngles"]["mean"][:,2]-self.m_normativeData["SpineAngles"]["sd"][:,2],
                        self.m_normativeData["SpineAngles"]["mean"][:,2]+self.m_normativeData["SpineAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "ThoraxAngles" in self.m_normativeData.keys():
                    self.fig.axes[6].fill_between(np.linspace(0,100,self.m_normativeData["ThoraxAngles"]["mean"].shape[0]),
                        self.m_normativeData["ThoraxAngles"]["mean"][:,0]-self.m_normativeData["ThoraxAngles"]["sd"][:,0],
                        self.m_normativeData["ThoraxAngles"]["mean"][:,0]+self.m_normativeData["ThoraxAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["ThoraxAngles"]["mean"].shape[0]),
                        self.m_normativeData["ThoraxAngles"]["mean"][:,1]-self.m_normativeData["ThoraxAngles"]["sd"][:,1],
                        self.m_normativeData["ThoraxAngles"]["mean"][:,1]+self.m_normativeData["ThoraxAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["ThoraxAngles"]["mean"].shape[0]),
                        self.m_normativeData["ThoraxAngles"]["mean"][:,2]-self.m_normativeData["ThoraxAngles"]["sd"][:,2],
                        self.m_normativeData["ThoraxAngles"]["mean"][:,2]+self.m_normativeData["ThoraxAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "NeckAngles" in self.m_normativeData.keys():
                    self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["NeckAngles"]["mean"].shape[0]),
                        self.m_normativeData["NeckAngles"]["mean"][:,0]-self.m_normativeData["NeckAngles"]["sd"][:,0],
                        self.m_normativeData["NeckAngles"]["mean"][:,0]+self.m_normativeData["NeckAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[10].fill_between(np.linspace(0,100,self.m_normativeData["NeckAngles"]["mean"].shape[0]),
                        self.m_normativeData["NeckAngles"]["mean"][:,1]-self.m_normativeData["NeckAngles"]["sd"][:,1],
                        self.m_normativeData["NeckAngles"]["mean"][:,1]+self.m_normativeData["NeckAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["NeckAngles"]["mean"].shape[0]),
                        self.m_normativeData["NeckAngles"]["mean"][:,2]-self.m_normativeData["NeckAngles"]["sd"][:,2],
                        self.m_normativeData["NeckAngles"]["mean"][:,2]+self.m_normativeData["NeckAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "HeadAngles" in self.m_normativeData.keys():
                    self.fig.axes[12].fill_between(np.linspace(0,100,self.m_normativeData["HeadAngles"]["mean"].shape[0]),
                        self.m_normativeData["HeadAngles"]["mean"][:,0]-self.m_normativeData["HeadAngles"]["sd"][:,0],
                        self.m_normativeData["HeadAngles"]["mean"][:,0]+self.m_normativeData["HeadAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[13].fill_between(np.linspace(0,100,self.m_normativeData["HeadAngles"]["mean"].shape[0]),
                        self.m_normativeData["HeadAngles"]["mean"][:,1]-self.m_normativeData["HeadAngles"]["sd"][:,1],
                        self.m_normativeData["HeadAngles"]["mean"][:,1]+self.m_normativeData["HeadAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[14].fill_between(np.linspace(0,100,self.m_normativeData["HeadAngles"]["mean"].shape[0]),
                        self.m_normativeData["HeadAngles"]["mean"][:,2]-self.m_normativeData["HeadAngles"]["sd"][:,2],
                        self.m_normativeData["HeadAngles"]["mean"][:,2]+self.m_normativeData["HeadAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

            elif self.m_bodyPart == enums.BodyPartPlot.UpperLimb:


                if "ShoulderAngles" in self.m_normativeData.keys():
                    self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["ShoulderAngles"]["mean"].shape[0]),
                        self.m_normativeData["ShoulderAngles"]["mean"][:,0]-self.m_normativeData["ShoulderAngles"]["sd"][:,0],
                        self.m_normativeData["ShoulderAngles"]["mean"][:,0]+self.m_normativeData["ShoulderAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["ShoulderAngles"]["mean"].shape[0]),
                        self.m_normativeData["ShoulderAngles"]["mean"][:,1]-self.m_normativeData["ShoulderAngles"]["sd"][:,1],
                        self.m_normativeData["ShoulderAngles"]["mean"][:,1]+self.m_normativeData["ShoulderAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[2].fill_between(np.linspace(0,100,self.m_normativeData["ShoulderAngles"]["mean"].shape[0]),
                        self.m_normativeData["ShoulderAngles"]["mean"][:,2]-self.m_normativeData["ShoulderAngles"]["sd"][:,2],
                        self.m_normativeData["ShoulderAngles"]["mean"][:,2]+self.m_normativeData["ShoulderAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

                if "ElbowAngles" in self.m_normativeData.keys():
                    self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["ElbowAngles"]["mean"].shape[0]),
                        self.m_normativeData["ElbowAngles"]["mean"][:,0]-self.m_normativeData["ElbowAngles"]["sd"][:,0],
                        self.m_normativeData["ElbowAngles"]["mean"][:,0]+self.m_normativeData["ElbowAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    # self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["ElbowAngles"]["mean"].shape[0]),
                    #     self.m_normativeData["ElbowAngles"]["mean"][:,1]-self.m_normativeData["ElbowAngles"]["sd"][:,1],
                    #     self.m_normativeData["ElbowAngles"]["mean"][:,1]+self.m_normativeData["ElbowAngles"]["sd"][:,1],
                    #     facecolor="green", alpha=0.5,linewidth=0)
                    #
                    # self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["ElbowAngles"]["mean"].shape[0]),
                    #     self.m_normativeData["ElbowAngles"]["mean"][:,2]-self.m_normativeData["ElbowAngles"]["sd"][:,2],
                    #     self.m_normativeData["ElbowAngles"]["mean"][:,2]+self.m_normativeData["ElbowAngles"]["sd"][:,2],
                    #     facecolor="green", alpha=0.5,linewidth=0)


                if "WristAngles" in self.m_normativeData.keys():
                    self.fig.axes[6].fill_between(np.linspace(0,100,self.m_normativeData["WristAngles"]["mean"].shape[0]),
                        self.m_normativeData["WristAngles"]["mean"][:,0]-self.m_normativeData["WristAngles"]["sd"][:,0],
                        self.m_normativeData["WristAngles"]["mean"][:,0]+self.m_normativeData["WristAngles"]["sd"][:,0],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["WristAngles"]["mean"].shape[0]),
                        self.m_normativeData["WristAngles"]["mean"][:,1]-self.m_normativeData["WristAngles"]["sd"][:,1],
                        self.m_normativeData["WristAngles"]["mean"][:,1]+self.m_normativeData["WristAngles"]["sd"][:,1],
                        facecolor="green", alpha=0.5,linewidth=0)

                    self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["WristAngles"]["mean"].shape[0]),
                        self.m_normativeData["WristAngles"]["mean"][:,2]-self.m_normativeData["WristAngles"]["sd"][:,2],
                        self.m_normativeData["WristAngles"]["mean"][:,2]+self.m_normativeData["WristAngles"]["sd"][:,2],
                        facecolor="green", alpha=0.5,linewidth=0)

        return self.fig

class TemporalKinematicsPlotViewer(AbstractPlotViewer):
    """ Plot temporal Kinematics

    Args:
        iAcq (btk.Acquisition): an acquisition
        pointLabelSuffix (str): suffix added model outputs
        bodyPart (enums.BodyPartPlot,Optional[enums.BodyPartPlot.LowerLimb]): body part

    """

    def __init__(self,iAcq,pointLabelSuffix=None,bodyPart=enums.BodyPartPlot.LowerLimb):

        super(TemporalKinematicsPlotViewer, self).__init__(iAcq)

        self.m_acq = self.m_input
        if isinstance(self.m_input,pyCGM2.btk.btkAcquisition):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a ma.Trial")


        self.m_bodyPart = bodyPart
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None


    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Time Kinematics \n """
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

        for ax in self.fig.axes:
            ax.set_ylabel("angle (deg)",size=8)
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

        ax12.set_xlabel("Cycle %",size=8)
        ax13.set_xlabel("Cycle %",size=8)
        ax14.set_xlabel("Cycle %",size=8)

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
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

            if not self.m_automaticYlim_flag:
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

        if self.m_bodyPart == enums.BodyPartPlot.Trunk:

            ax0.set_title("Pelvis Tilt" ,size=8)
            ax1.set_title("Pelvis Obliquity" ,size=8)
            ax2.set_title("Pelvis Rotation" ,size=8)
            ax3.set_title("Spine Flexion" ,size=8)
            ax4.set_title("Spine Adduction" ,size=8)
            ax5.set_title("Spine Rotation" ,size=8)
            ax6.set_title("Thorax Anteversion" ,size=8)
            ax7.set_title("Thorax Obliquity" ,size=8)
            ax8.set_title("Thorax Rotation" ,size=8)

            ax9.set_title("Neck Forward Tilt" ,size=8)
            ax10.set_title("Neck Lateral Tilt" ,size=8)
            ax11.set_title("Neck Rotation" ,size=8)

            ax12.set_title("Head Forward Tilt" ,size=8)
            ax13.set_title("Head Lateral Tilt" ,size=8)
            ax14.set_title("Head Rotation" ,size=8)

            if not self.m_automaticYlim_flag:
                ax0.set_ylim([0,60])
                ax1.set_ylim([-30,30])
                ax2.set_ylim([-30,30])

                ax3.set_ylim( [-30,30])
                ax4.set_ylim([-30,30])
                ax5.set_ylim([-30,30])

                ax6.set_ylim([-30,30])
                ax7.set_ylim([-30,30])
                ax8.set_ylim([-30,30])

        if self.m_bodyPart == enums.BodyPartPlot.UpperLimb:

            ax0.set_title("Shoulder flexion" ,size=8)
            ax1.set_title("Shoulder Adduction" ,size=8)
            ax2.set_title("Shoulder Rotation" ,size=8)
            ax3.set_title("Elbow Flexion" ,size=8)
            # ax4.set_title("Elbow Adduction" ,size=8)
            # ax5.set_title("Spine Rotation" ,size=8)
            ax6.set_title("Ulnar Deviation" ,size=8)
            ax7.set_title("Wrist Extension" ,size=8)
            ax8.set_title("Wrist Rotation" ,size=8)

            if not self.m_automaticYlim_flag:
                ax0.set_ylim([-60,60])
                ax1.set_ylim([-30,30])
                ax2.set_ylim([-30,30])

                ax3.set_ylim( [-60,60])
                ax4.set_ylim([-30,30])
                ax5.set_ylim([-30,30])

                ax6.set_ylim([-30,30])
                ax7.set_ylim([-30,30])
                ax8.set_ylim([0,160])

                ax9.set_ylim([-30,30])
                ax10.set_ylim([-30,30])
                ax11.set_ylim([-30,30])

                ax12.set_ylim([-50,30])
                ax13.set_ylim([-30,30])
                ax14.set_ylim([-30,30])


    def __setData(self):


        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "LPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "LPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "LPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "LHipAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "LHipAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "LHipAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "LKneeAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "LKneeAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "LKneeAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "LAnkleAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "LAnkleAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "LFootProgressAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[12],self.m_acq,
                                    "LForeFootAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[13],self.m_acq,
                                    "LForeFootAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[14],self.m_acq,
                                    "LForeFootAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")


            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "RPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "RPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "RPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "RHipAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "RHipAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "RHipAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "RKneeAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "RKneeAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "RKneeAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "RAnkleAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "RAnkleAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "RFootProgressAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[12],self.m_acq,
                                    "RForeFootAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[13],self.m_acq,
                                    "RForeFootAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[14],self.m_acq,
                                    "RForeFootAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

        if self.m_bodyPart == enums.BodyPartPlot.Trunk:

            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "LPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "LPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "LPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "LSpineAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "LSpineAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "LSpineAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "LThoraxAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "LThoraxAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "LThoraxAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")

            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "LNeckAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "LNeckAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "LNeckAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[12],self.m_acq,
                                    "LHeadAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[13],self.m_acq,
                                    "LHeadAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[14],self.m_acq,
                                    "LHeadAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")



            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "RPelvisAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                                    "RPelvisAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "RPelvisAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "RSpineAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "RSpineAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "RSpineAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "RThoraxAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "RThoraxAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "RThoraxAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "RNeckAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "RNeckAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "RNeckAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[12],self.m_acq,
                                    "RHeadAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[13],self.m_acq,
                                    "RHeadAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[14],self.m_acq,
                                    "RHeadAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")


        if self.m_bodyPart == enums.BodyPartPlot.UpperLimb:

            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "LShoulderAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "LShoulderAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "LShoulderAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "LElbowAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            # plot.temporalPlot(self.fig.axes[4],self.m_acq,
            #                         "LElbowAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            # plot.temporalPlot(self.fig.axes[5],self.m_acq,
            #                         "LElbowAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "LWristAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "LWristAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "LWristAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")


            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "RShoulderAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                                    "RShoulderAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "RShoulderAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "RElbowAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            # plot.temporalPlot(self.fig.axes[4],self.m_acq,
            #                         "RElbowAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            # plot.temporalPlot(self.fig.axes[5],self.m_acq,
            #                         "RElbowAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "RWristAngles",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "RWristAngles",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "RWristAngles",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")



    def plotPanel(self):
        """Plot the panel"""

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig


class NormalizedKineticsPlotViewer(AbstractPlotViewer):
    """ Plot time-Normalized kinetics

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis instance`
        pointLabelSuffix (str): suffix added model outputs
        bodyPart (enums.BodyPartPlot,Optional[enums.BodyPartPlot.LowerLimb]): body part

    """

    def __init__(self,iAnalysis,pointLabelSuffix=None,bodyPart=enums.BodyPartPlot.LowerLimb):

        super(NormalizedKineticsPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_bodyPart = bodyPart
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_concretePlotFunction = None

    def setConcretePlotFunction(self, concreteplotFunction):
        """set a plot function ( see `plot`)

        Args:
            concreteplotFunction (function): plot function

        """
        self.m_concretePlotFunction = concreteplotFunction

    def __setLayer(self):

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        if self.m_concretePlotFunction.__name__ in ["descriptivePlot","gaitDescriptivePlot"]:
            title=u""" Descriptive Time-normalized Kinetics \n """
        elif self.m_concretePlotFunction.__name__ in ["consistencyPlot","gaitConsistencyPlot"]:
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

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

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

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:

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

            if not self.m_automaticYlim_flag:
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
        else:
            raise Exception("Plot panel not implemented yet")


    def setNormativeDataset(self,iNormativeDataSet):
        """ Set the normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): a normative dataset instance
        """

        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
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

    def plotPanel(self):
        """Plot the panel"""

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        if self.m_normativeData is not None:
            if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
                    self.m_normativeData["HipMoment"]["mean"][:,0]-self.m_normativeData["HipMoment"]["sd"][:,0],
                    self.m_normativeData["HipMoment"]["mean"][:,0]+self.m_normativeData["HipMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
                    self.m_normativeData["HipMoment"]["mean"][:,1]-self.m_normativeData["HipMoment"]["sd"][:,1],
                    self.m_normativeData["HipMoment"]["mean"][:,1]+self.m_normativeData["HipMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["HipPower"]["mean"].shape[0]),
                    self.m_normativeData["HipPower"]["mean"][:,2]-self.m_normativeData["HipPower"]["sd"][:,2],
                    self.m_normativeData["HipPower"]["mean"][:,2]+self.m_normativeData["HipPower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)


                self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
                    self.m_normativeData["KneeMoment"]["mean"][:,0]-self.m_normativeData["KneeMoment"]["sd"][:,0],
                    self.m_normativeData["KneeMoment"]["mean"][:,0]+self.m_normativeData["KneeMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
                    self.m_normativeData["KneeMoment"]["mean"][:,1]-self.m_normativeData["KneeMoment"]["sd"][:,1],
                    self.m_normativeData["KneeMoment"]["mean"][:,1]+self.m_normativeData["KneeMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["KneePower"]["mean"].shape[0]),
                    self.m_normativeData["KneePower"]["mean"][:,2]-self.m_normativeData["KneePower"]["sd"][:,2],
                    self.m_normativeData["KneePower"]["mean"][:,2]+self.m_normativeData["KneePower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)



                self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
                    self.m_normativeData["AnkleMoment"]["mean"][:,0]-self.m_normativeData["AnkleMoment"]["sd"][:,0],
                    self.m_normativeData["AnkleMoment"]["mean"][:,0]+self.m_normativeData["AnkleMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
                    self.m_normativeData["AnkleMoment"]["mean"][:,1]-self.m_normativeData["AnkleMoment"]["sd"][:,1],
                    self.m_normativeData["AnkleMoment"]["mean"][:,1]+self.m_normativeData["AnkleMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)


                self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["AnklePower"]["mean"].shape[0]),
                    self.m_normativeData["AnklePower"]["mean"][:,2]-self.m_normativeData["AnklePower"]["sd"][:,2],
                    self.m_normativeData["AnklePower"]["mean"][:,2]+self.m_normativeData["AnklePower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)

        return self.fig





class TemporalKineticsPlotViewer(AbstractPlotViewer):
    """ Plot temporal Kinetics

    Args:
        iAcq (btk.Acquisition): an acquisition
        pointLabelSuffix (str): suffix added model outputs
        bodyPart (enums.BodyPartPlot,Optional[enums.BodyPartPlot.LowerLimb]): body part

    """

    def __init__(self,iAcq,pointLabelSuffix=None,bodyPart=enums.BodyPartPlot.LowerLimb):

        super(TemporalKineticsPlotViewer, self).__init__(iAcq)

        self.m_acq = self.m_input
        if isinstance(self.m_input,pyCGM2.btk.btkAcquisition):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a ma.Trial")

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None

        self.m_bodyPart = bodyPart

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

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)


        for ax in [self.fig.axes[0],self.fig.axes[1],self.fig.axes[2],
                   self.fig.axes[4],self.fig.axes[5],self.fig.axes[0],
                   self.fig.axes[8],self.fig.axes[9],self.fig.axes[10]]:
            ax.set_ylabel("moment (N.mm.kg-1)",size=8)

        for ax in [self.fig.axes[3],self.fig.axes[7],self.fig.axes[8]]:
            ax.set_ylabel("power (W.Kg-1)",size=8)

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
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



            ax9.set_xlabel("Frame %",size=8)
            ax10.set_xlabel("Frame %",size=8)
            ax11.set_xlabel("Frame %",size=8)

            if not self.m_automaticYlim_flag:
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
        else:
            raise Exception ("Plot Panel not implemented yet")


    def __setData(self):

        if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "LHipMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "LHipMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "LHipMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "LHipPower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")

            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "LKneeMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "LKneeMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "LKneeMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "LKneePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")

            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "LAnkleMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "LAnkleMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "LAnkleMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "LAnklePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="red")


            plot.temporalPlot(self.fig.axes[0],self.m_acq,
                                    "RHipMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[1],self.m_acq,
                            "RHipMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[2],self.m_acq,
                                    "RHipMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[3],self.m_acq,
                                    "RHipPower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

            plot.temporalPlot(self.fig.axes[4],self.m_acq,
                                    "RKneeMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[5],self.m_acq,
                                    "RKneeMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[6],self.m_acq,
                                    "RKneeMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[7],self.m_acq,
                                    "RKneePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")

            plot.temporalPlot(self.fig.axes[8],self.m_acq,
                                    "RAnkleMoment",0,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[9],self.m_acq,
                                    "RAnkleMoment",1,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[10],self.m_acq,
                                    "RAnkleMoment",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")
            plot.temporalPlot(self.fig.axes[11],self.m_acq,
                                    "RAnklePower",2,pointLabelSuffix=self.m_pointLabelSuffix,color="blue")


    def plotPanel(self):
        """plot the panel"""

        self.__setLayer()
        self.__setData()

        # normative dataset not implemented

        return self.fig
