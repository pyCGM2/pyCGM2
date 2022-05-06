# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

"""
Module contains `plotViewers` for comparing data from different `analysis` instances

"""

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt

# pyCGM2
import pyCGM2
from pyCGM2 import enums
from pyCGM2.Report.Viewers import plotViewers





class KinematicsPlotComparisonViewer(plotViewers.AbstractPlotViewer):
    """ Compare kinematics

    Args:
        iAnalyses (list ): `pyCGM2.Processing.analysis.Analysis` instances
        context (str): event context
        pointLabelSuffix (str) - suffix added to model outputs
        legends(list): labels caracterizing the `analysis` instances
        pointLabelSuffix_lst(list,Optional[None]): suffix added to model outputs of the `analysis` instances
        bodyPart(pyCGM2.enums.BodyPartPlot,Optional[pyCGM2.enums.BodyPartPlot.LowerLimb]):body part

    """


    def __init__(self,iAnalyses,context,legends,pointLabelSuffix_lst=None,bodyPart=enums.BodyPartPlot.LowerLimb):

        super(KinematicsPlotComparisonViewer, self).__init__(iAnalyses)


        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input
        self.m_context = context
        self.m_pointLabelSuffixes = pointLabelSuffix_lst
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_legends = legends
        self.m_bodyPart = bodyPart

        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")
        if pointLabelSuffix_lst is not None:
            if len(iAnalyses) != len(pointLabelSuffix_lst):
                raise Exception("list of point label suffix don t match analysis. Must have same length")

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

            for ax in self.fig.axes:
                ax.set_ylabel("angle (deg)",size=8)

            ax12.set_xlabel("Cycle %",size=8)
            ax13.set_xlabel("Cycle %",size=8)
            ax14.set_xlabel("Cycle %",size=8)

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
        else:
            LOGGER.logger.warning("Plot Panel not implemented yet")


    def __setLegend(self,axisIndex):
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet):
        """Set a normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance

        """

        # iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):


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

                if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    # hip
                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                            "LHipAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    # knee
                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "LKneeAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    # ankle
                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kinematicStats,
                            "LAnkleAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)

                    # foot progress
                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kinematicStats,
                            "LFootProgressAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    #forefoot
                    self.m_concretePlotFunction(self.fig.axes[12],analysis.kinematicStats,
                            "LForeFootAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[13],analysis.kinematicStats,
                            "LForeFootAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[14],analysis.kinematicStats,
                            "LForeFootAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

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

                if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    # hip
                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                            "RHipAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    # knee
                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "RKneeAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    # ankle
                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kinematicStats,
                            "RAnkleAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)

                    # foot progress
                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kinematicStats,
                            "RFootProgressAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)
            #
                    # foreFoot
                    self.m_concretePlotFunction(self.fig.axes[12],analysis.kinematicStats,
                            "RForeFootAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[13],analysis.kinematicStats,
                            "RForeFootAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)
                    self.m_concretePlotFunction(self.fig.axes[14],analysis.kinematicStats,
                            "RForeFootAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                i+=1

    def plotPanel(self):
        """ plot the panel"""


        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")


        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

        if self.m_normativeData is not None:
            if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                self.fig.axes[0].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["PelvisAngles"]["mean"][:,0]-self.m_normativeData["PelvisAngles"]["sd"][:,0],
                    self.m_normativeData["PelvisAngles"]["mean"][:,0]+self.m_normativeData["PelvisAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[1].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["PelvisAngles"]["mean"][:,1]-self.m_normativeData["PelvisAngles"]["sd"][:,1],
                    self.m_normativeData["PelvisAngles"]["mean"][:,1]+self.m_normativeData["PelvisAngles"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[2].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["PelvisAngles"]["mean"][:,2]-self.m_normativeData["PelvisAngles"]["sd"][:,2],
                    self.m_normativeData["PelvisAngles"]["mean"][:,2]+self.m_normativeData["PelvisAngles"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)


                self.fig.axes[3].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipAngles"]["mean"][:,0]-self.m_normativeData["HipAngles"]["sd"][:,0],
                    self.m_normativeData["HipAngles"]["mean"][:,0]+self.m_normativeData["HipAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[4].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipAngles"]["mean"][:,1]-self.m_normativeData["HipAngles"]["sd"][:,1],
                    self.m_normativeData["HipAngles"]["mean"][:,1]+self.m_normativeData["HipAngles"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[5].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipAngles"]["mean"][:,2]-self.m_normativeData["HipAngles"]["sd"][:,2],
                    self.m_normativeData["HipAngles"]["mean"][:,2]+self.m_normativeData["HipAngles"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[6].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["KneeAngles"]["mean"][:,0]-self.m_normativeData["KneeAngles"]["sd"][:,0],
                    self.m_normativeData["KneeAngles"]["mean"][:,0]+self.m_normativeData["KneeAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[9].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["AnkleAngles"]["mean"][:,0]-self.m_normativeData["AnkleAngles"]["sd"][:,0],
                    self.m_normativeData["AnkleAngles"]["mean"][:,0]+self.m_normativeData["AnkleAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[11].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["FootProgressAngles"]["mean"][:,2]-self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                    self.m_normativeData["FootProgressAngles"]["mean"][:,2]+self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)

        return self.fig

class KineticsPlotComparisonViewer(plotViewers.AbstractPlotViewer):
    """ Compare kinetics

    Args:
        iAnalyses (list ):  `pyCGM2.Processing.analysis.Analysis` instances
        context (str): event context
        pointLabelSuffix (str) - suffix added to model outputs
        legends(list): labels caracterizing the `analysis` instances
        pointLabelSuffix_lst(list,Optional[None]): suffix added to model outputs of the `analysis` instances
        bodyPart(pyCGM2.enums.BodyPartPlot,Optional[pyCGM2.enums.BodyPartPlot.LowerLimb]):body part

    """

    def __init__(self,iAnalyses,context,legends,pointLabelSuffix_lst=None,bodyPart= enums.BodyPartPlot.LowerLimb):


        super(KineticsPlotComparisonViewer, self).__init__(iAnalyses)

        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input
        self.m_context = context
        self.m_pointLabelSuffixes = pointLabelSuffix_lst
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_legends = legends
        self.m_bodyPart=bodyPart

        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")
        if pointLabelSuffix_lst is not None:
            if len(iAnalyses) != len(pointLabelSuffix_lst):
                raise Exception("list of point label suffix don t match analysis. Must have same length")

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

            for ax in [self.fig.axes[0],self.fig.axes[1],self.fig.axes[2],
                       self.fig.axes[4],self.fig.axes[5],self.fig.axes[0],
                       self.fig.axes[8],self.fig.axes[9],self.fig.axes[10]]:
                ax.set_ylabel("moment (N.mm.kg-1)",size=8)

            for ax in [self.fig.axes[3],self.fig.axes[7],self.fig.axes[8]]:
                ax.set_ylabel("power (W.Kg-1)",size=8)

            ax8.set_xlabel("Cycle %",size=8)
            ax9.set_xlabel("Cycle %",size=8)
            ax10.set_xlabel("Cycle %",size=8)
            ax11.set_xlabel("Cycle %",size=8)

            if self.m_automaticYlim_flag:
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
            LOGGER.logger.warning("Plot Panel not implemented yet")

    def __setLegend(self,axisIndex):
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet):
        """Set a normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance

        """
        # iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):

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

                if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                    # hip
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                            legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)


                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kineticStats,
                            "LHipMoment"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kineticStats,
                            "LHipPower"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    # knee
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kineticStats,
                            "LKneeMoment"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kineticStats,
                            "LKneePower"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    # ankle
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kineticStats,
                            "LAnkleMoment"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kineticStats,
                            "LAnklePower"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None)
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

                if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                    # hip
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                            legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)


                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kineticStats,
                            "RHipMoment"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kineticStats,
                            "RHipPower"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    # knee
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kineticStats,
                            "RKneeMoment"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kineticStats,
                            "RKneePower"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    # ankle
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kineticStats,
                            "RAnkleMoment"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kineticStats,
                            "RAnklePower"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None)

                i+=1

        #
    def plotPanel(self):
        """ plot the panel"""

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

        if self.m_normativeData is not None:
            if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
                self.fig.axes[0].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipMoment"]["mean"][:,0]-self.m_normativeData["HipMoment"]["sd"][:,0],
                    self.m_normativeData["HipMoment"]["mean"][:,0]+self.m_normativeData["HipMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[1].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipMoment"]["mean"][:,1]-self.m_normativeData["HipMoment"]["sd"][:,1],
                    self.m_normativeData["HipMoment"]["mean"][:,1]+self.m_normativeData["HipMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[3].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["HipPower"]["mean"][:,2]-self.m_normativeData["HipPower"]["sd"][:,2],
                    self.m_normativeData["HipPower"]["mean"][:,2]+self.m_normativeData["HipPower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)


                self.fig.axes[4].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["KneeMoment"]["mean"][:,0]-self.m_normativeData["KneeMoment"]["sd"][:,0],
                    self.m_normativeData["KneeMoment"]["mean"][:,0]+self.m_normativeData["KneeMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[5].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["KneeMoment"]["mean"][:,1]-self.m_normativeData["KneeMoment"]["sd"][:,1],
                    self.m_normativeData["KneeMoment"]["mean"][:,1]+self.m_normativeData["KneeMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[7].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["KneePower"]["mean"][:,2]-self.m_normativeData["KneePower"]["sd"][:,2],
                    self.m_normativeData["KneePower"]["mean"][:,2]+self.m_normativeData["KneePower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)



                self.fig.axes[8].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["AnkleMoment"]["mean"][:,0]-self.m_normativeData["AnkleMoment"]["sd"][:,0],
                    self.m_normativeData["AnkleMoment"]["mean"][:,0]+self.m_normativeData["AnkleMoment"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[9].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["AnkleMoment"]["mean"][:,1]-self.m_normativeData["AnkleMoment"]["sd"][:,1],
                    self.m_normativeData["AnkleMoment"]["mean"][:,1]+self.m_normativeData["AnkleMoment"]["sd"][:,1],
                    facecolor="green", alpha=0.5,linewidth=0)


                self.fig.axes[11].fill_between(np.linspace(0,100,51),
                    self.m_normativeData["AnklePower"]["mean"][:,2]-self.m_normativeData["AnklePower"]["sd"][:,2],
                    self.m_normativeData["AnklePower"]["mean"][:,2]+self.m_normativeData["AnklePower"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)

        return self.fig
