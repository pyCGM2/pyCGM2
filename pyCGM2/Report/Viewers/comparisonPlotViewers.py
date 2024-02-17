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

from pyCGM2.Processing.analysis import Analysis


from pyCGM2.Report.normativeDatasets import NormativeData
from typing import List, Tuple, Dict, Optional, Union, Callable



class KinematicsPlotComparisonViewer(plotViewers.PlotViewer):
    """
    A viewer for comparing kinematic data across multiple `Analysis` instances.
    
    This class allows for the visualization of kinematic data, enabling a comparison
    between different analyses. It supports customization for specific body parts and
    accommodates various data suffixes and contexts.

    Args:
        iAnalyses (List[Analysis]): Instances of analyses to be compared.
        context (str): The context of the biomechanical event.
        legends (List[str]): Descriptive labels for each analysis instance.
        pointLabelSuffix_lst (Optional[List[str]]): Suffixes for model output labels.
        bodyPart (enums.BodyPartPlot): The body part to be visualized, defaults to LowerLimb.

    """


    def __init__(self,iAnalyses:List[Analysis],context:List[str],legends:List[str],pointLabelSuffix_lst:Optional[List[str]]=None,
                 bodyPart:enums.BodyPartPlot=enums.BodyPartPlot.LowerLimb):
        """Initialize the comparison plot viewer"""


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

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Sets a specific plot function for kinematic data visualization.
        
        This method allows the customization of the plot based on the user's needs,
        enabling the use of various plot types from the pyCGM2.Report.plot library.

        Args:
            concreteplotFunction (Callable): A plot function from pyCGM2.Report.plot.
        """

        self.m_concretePlotFunction = concreteplotFunction

    def __setLayer(self):

        """
        Private method to set up the plot layers.
        
        This method initializes the figure and axes for the plot, setting titles,
        subtitles, and adjusting layout parameters. It also configures the axes
        based on the selected body part.

        Note:
            This method is internally used by the class and not intended for external use.
        """
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

            if self.m_automaticYlim_flag:
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
        else:
            LOGGER.logger.warning("Plot Panel not implemented yet")


    def __setLegend(self,axisIndex):
        """
        Private method to set the legend for the plot.
        
        Configures and places the legend on the plot based on the provided axis index.

        Args:
            axisIndex (int): Index of the axis where the legend is to be placed.

        Note:
            This method is internally used by the class and not intended for external use.
        """
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Sets the normative dataset for comparison with the analysis data.

        Args:
            iNormativeDataSet (NormativeData): An instance of normative data for comparison.
        """

        # iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        """
        Private method to set the data for plotting.
        
        Organizes and prepares the kinematic data from the analysis instances for plotting.

        Note:
            This method is internally used by the class and not intended for external use.
        """

        if self.m_context == "Left":
            colormap = plt.cm.Reds
            colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]
            colormap_i = [(0,0,0)] + colormap_i

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
                
                elif self.m_bodyPart == enums.BodyPartPlot.Trunk:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "LPelvisAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "LSpineAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                            "LSpineAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                            "LSpineAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "LThoraxAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "LThoraxAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "LThoraxAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kinematicStats,
                            "LNeckAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kinematicStats,
                            "LNeckAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kinematicStats,
                            "LNeckAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                    self.m_concretePlotFunction(self.fig.axes[12],analysis.kinematicStats,
                            "LHeadAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[13],analysis.kinematicStats,
                            "LHeadAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[14],analysis.kinematicStats,
                            "LHeadAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)



                elif self.m_bodyPart == enums.BodyPartPlot.UpperLimb:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "LShoulderAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "LShoulderAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "LShoulderAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "LElbowAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    # self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                    #         "LElbowAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    # legendLabel=legend)
                    # self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                    #         "LElbowAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    # legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "LWristAngles"+suffixPlus,"Left",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "LWristAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "LWristAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                i+=1

        if self.m_context == "Right":
            colormap = plt.cm.Blues
            colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]
            colormap_i = [(0,0,0)] + colormap_i

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

                elif self.m_bodyPart == enums.BodyPartPlot.Trunk:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "RPelvisAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "RSpineAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                            "RSpineAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                            "RSpineAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "RThoraxAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "RThoraxAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "RThoraxAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                    self.m_concretePlotFunction(self.fig.axes[9],analysis.kinematicStats,
                            "RNeckAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[10],analysis.kinematicStats,
                            "RNeckAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[11],analysis.kinematicStats,
                            "RNeckAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                    self.m_concretePlotFunction(self.fig.axes[12],analysis.kinematicStats,
                            "RHeadAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[13],analysis.kinematicStats,
                            "RHeadAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[14],analysis.kinematicStats,
                            "RHeadAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                elif self.m_bodyPart == enums.BodyPartPlot.UpperLimb:
                    self.m_concretePlotFunction(self.fig.axes[0],analysis.kinematicStats,
                            "RShoulderAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[1],analysis.kinematicStats,
                            "RShoulderAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[2],analysis.kinematicStats,
                            "RShoulderAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[3],analysis.kinematicStats,
                            "RElbowAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    # self.m_concretePlotFunction(self.fig.axes[4],analysis.kinematicStats,
                    #         "LElbowAngles"+suffixPlus,"Left",1, color=colormap_i[i], customLimits=None,
                    # legendLabel=legend)
                    # self.m_concretePlotFunction(self.fig.axes[5],analysis.kinematicStats,
                    #         "LElbowAngles"+suffixPlus,"Left",2, color=colormap_i[i], customLimits=None,
                    # legendLabel=legend)

                    self.m_concretePlotFunction(self.fig.axes[6],analysis.kinematicStats,
                            "RWristAngles"+suffixPlus,"Right",0, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[7],analysis.kinematicStats,
                            "RWristAngles"+suffixPlus,"Right",1, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)
                    self.m_concretePlotFunction(self.fig.axes[8],analysis.kinematicStats,
                            "RWristAngles"+suffixPlus,"Right",2, color=colormap_i[i], customLimits=None,
                    legendLabel=legend)


                i+=1

    def plotPanel(self):
        """
        Generates and plots the kinematic comparison panel.

        This method orchestrates the plotting process, including setting up the layers,
        arranging data, and rendering the final plot.

        Returns:
            matplotlib.figure.Figure: The generated plot as a matplotlib figure object.
            
        Raises:
            Exception: If the concrete plot function is not defined.
        """


        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")


        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

        if self.m_normativeData is not None:
            if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
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

                self.fig.axes[6].fill_between(np.linspace(0,100,self.m_normativeData["KneeAngles"]["mean"].shape[0]),
                    self.m_normativeData["KneeAngles"]["mean"][:,0]-self.m_normativeData["KneeAngles"]["sd"][:,0],
                    self.m_normativeData["KneeAngles"]["mean"][:,0]+self.m_normativeData["KneeAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["AnkleAngles"]["mean"].shape[0]),
                    self.m_normativeData["AnkleAngles"]["mean"][:,0]-self.m_normativeData["AnkleAngles"]["sd"][:,0],
                    self.m_normativeData["AnkleAngles"]["mean"][:,0]+self.m_normativeData["AnkleAngles"]["sd"][:,0],
                    facecolor="green", alpha=0.5,linewidth=0)

                self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["FootProgressAngles"]["mean"].shape[0]),
                    self.m_normativeData["FootProgressAngles"]["mean"][:,2]-self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                    self.m_normativeData["FootProgressAngles"]["mean"][:,2]+self.m_normativeData["FootProgressAngles"]["sd"][:,2],
                    facecolor="green", alpha=0.5,linewidth=0)

        return self.fig

class KineticsPlotComparisonViewer(plotViewers.PlotViewer):
    """ 
    A viewer for comparing kinetic data across multiple `Analysis` instances.
    
    Similar to kineticsPlotComparisonViewer, this class is designed for the visualization
    and comparison of kinetic data from different analysis instances. It supports kinetic
    data visualization for specified body parts and handles various data suffixes and contexts.


    Args:
        iAnalyses (List[Analysis]): Instances of analyses to be compared.
        context (str): The context of the biomechanical event.
        legends (List[str]): Descriptive labels for each analysis instance.
        pointLabelSuffix_lst (Optional[List[str]]): Suffixes for model output labels.
        bodyPart (enums.BodyPartPlot): The body part to be visualized, defaults to LowerLimb.


    """

    def __init__(self,iAnalyses:List[Analysis],context:List[str],legends:List[str],pointLabelSuffix_lst:Optional[List[str]]=None,
                 bodyPart:enums.BodyPartPlot= enums.BodyPartPlot.LowerLimb):
        """Initialize the comparison plot viewer"""


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

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Sets a specific plot function for kinetic data visualization.
        
        This method allows the customization of the plot based on the user's needs,
        enabling the use of various plot types from the pyCGM2.Report.plot library.

        Args:
            concreteplotFunction (Callable): A plot function from pyCGM2.Report.plot.
        """
        self.m_concretePlotFunction = concreteplotFunction

    def __setLayer(self):
        """
        Private method to set up the plot layers.
        
        This method initializes the figure and axes for the plot, setting titles,
        subtitles, and adjusting layout parameters. It also configures the axes
        based on the selected body part.

        Note:
            This method is internally used by the class and not intended for external use.
        """

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
        """
        Private method to set the legend for the plot.
        
        Configures and places the legend on the plot based on the provided axis index.

        Args:
            axisIndex (int): Index of the axis where the legend is to be placed.

        Note:
            This method is internally used by the class and not intended for external use.
        """
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Sets the normative dataset for comparison with the analysis data.

        Args:
            iNormativeDataSet (NormativeData): An instance of normative data for comparison.
        """
        # iNormativeDataSet.constructNormativeData()
        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        """
        Private method to set the data for plotting.
        
        Organizes and prepares the kinematic data from the analysis instances for plotting.

        Note:
            This method is internally used by the class and not intended for external use.
        """
        if self.m_context == "Left":
            colormap = plt.cm.Reds
            colormap_i=[colormap(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]
            colormap_i = [(0,0,0)] + colormap_i

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
            colormap_i = [(0,0,0)] + colormap_i

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
        """
        Generates and plots the kinematic comparison panel.

        This method orchestrates the plotting process, including setting up the layers,
        arranging data, and rendering the final plot.

        Returns:
            matplotlib.figure.Figure: The generated plot as a matplotlib figure object.
            
        Raises:
            Exception: If the concrete plot function is not defined.
        """

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()
        self.__setLegend(0)

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

class GroundReactionForceComparisonViewer(plotViewers.PlotViewer):
    """ 
    A viewer for comparing Ground reaction Force  across multiple `Analysis` instances.
    
    This class is designed for the visualization
    and comparison of ground reaction force from different analysis instances.


    Args:
        iAnalyses (List[Analysis]): Instances of analyses to be compared.
        legends (List[str]): Descriptive labels for each analysis instance.
        pointLabelSuffix_lst (Optional[List[str]]): Suffixes for model output labels.
        bodyPart (enums.BodyPartPlot): The body part to be visualized, defaults to LowerLimb.


    """

    def __init__(self,iAnalyses:List[Analysis],legends:List[str],pointLabelSuffix_lst:Optional[List[str]]=None):
        """Initialize the comparison plot viewer"""


        super(GroundReactionForceComparisonViewer, self).__init__(iAnalyses)

        for itAnalysis in iAnalyses:
            if isinstance(itAnalysis,pyCGM2.Processing.analysis.Analysis):
                pass
            else:
                LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_analysis = self.m_input
        self.m_pointLabelSuffixes = pointLabelSuffix_lst
        self.m_normativeData = None
        self.m_flagConsistencyOnly = False
        self.m_legends = legends

        if len(iAnalyses) != len(legends):
            raise Exception("legends don t match analysis. Must have same length")
        if pointLabelSuffix_lst is not None:
            if len(iAnalyses) != len(pointLabelSuffix_lst):
                raise Exception("list of point label suffix don t match analysis. Must have same length")

        self.m_concretePlotFunction = None

    def setConcretePlotFunction(self, concreteplotFunction:Callable):
        """
        Sets a specific plot function for kinetic data visualization.
        
        This method allows the customization of the plot based on the user's needs,
        enabling the use of various plot types from the pyCGM2.Report.plot library.

        Args:
            concreteplotFunction (Callable): A plot function from pyCGM2.Report.plot.
        """
        self.m_concretePlotFunction = concreteplotFunction

    def __setLayer(self):
        """
        private method to Set up the plot layers for ground reaction force visualization.
        """

        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        if self.m_concretePlotFunction.__name__ in ["descriptivePlot","gaitDescriptivePlot"]:
            title=u""" Descriptive Time-normalized Ground reaction force \n """
        elif self.m_concretePlotFunction.__name__ in ["consistencyPlot","gaitConsistencyPlot"]:
            title=u""" Consistency Time-normalized Ground reaction force \n """
        else :
            title=u"""\n"""


        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        nrow = 3 

        ax0 = plt.subplot(nrow,3,1)# 
        ax1 = plt.subplot(nrow,3,2)# 
        ax2 = plt.subplot(nrow,3,3)# 

        ax3 = plt.subplot(nrow,3,4)# 
        ax4 = plt.subplot(nrow,3,5)# 
        ax5 = plt.subplot(nrow,3,6)#

        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)

        #  GRF
        for ax in [self.fig.axes[0],self.fig.axes[1],self.fig.axes[2]]:
            ax.set_ylabel("Ground reaction force (N.kg-1)",size=8)

        self.fig.axes[0].set_title("Longitudinal Force" ,size=8)
        self.fig.axes[1].set_title("Lateral Force" ,size=8)
        self.fig.axes[2].set_title("Vertical Force" ,size=8)

        self.fig.axes[2].axhline(9.81,color="black",ls='dashed')

        self.fig.axes[3].set_title("Longitudinal Force" ,size=8)
        self.fig.axes[4].set_title("Lateral Force" ,size=8)
        self.fig.axes[5].set_title("Vertical Force" ,size=8)

        self.fig.axes[5].axhline(9.81,color="black",ls='dashed')


        if not self.m_automaticYlim_flag:
            pass
       
        for ax in self.fig.axes[0:6]:    
            ax.set_xlabel("Cycle %",size=8)
            ax.set_xlabel("Cycle %",size=8)
            ax.set_xlabel("Cycle %",size=8)

    def __setLegend(self,axisIndex):
        """
        Private method to set the legend for the plot.
        
        Configures and places the legend on the plot based on the provided axis index.

        Args:
            axisIndex (int): Index of the axis where the legend is to be placed.

        Note:
            This method is internally used by the class and not intended for external use.
        """
        self.fig.axes[axisIndex].legend(fontsize=6)
        #self.fig.axes[axisIndex].legend(fontsize=6, bbox_to_anchor=(0,1.2,1,0.2), loc="lower left",
        #    mode="None", borderaxespad=0, ncol=len(self.m_analysis))

    def setNormativeDataset(self,iNormativeDataSet:NormativeData):
        """
        Set the normative dataset for comparison.

        Args:
            iNormativeDataSet (NormativeData): An instance of a normative dataset.
        """

        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        """
        Set the data for ground reaction force visualization.
        """
        colormap_l = plt.cm.Reds
        colormap_i_l=[colormap_l(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]
        colormap_i_l = [(0,0,0)] + colormap_i_l

        colormap_r = plt.cm.Blues
        colormap_i_r=[colormap_r(i) for i in np.linspace(0.2, 1, len(self.m_analysis))]
        colormap_i_r = [(0,0,0)] + colormap_i_r

        i = 0
        for analysis in self.m_analysis:
            if self.m_pointLabelSuffixes is not None:
                suffixPlus = "_" + self.m_pointLabelSuffixes[i] if self.m_pointLabelSuffixes[i] !="" else ""
            else:
                suffixPlus=""

            legend= self.m_legends[i]

        
            self.m_concretePlotFunction(self.fig.axes[0],analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",0, color=colormap_i_l[i], customLimits=None,legendLabel=legend)

            self.m_concretePlotFunction(self.fig.axes[1],analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",1, color=colormap_i_l[i], customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[2],analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",2, color=colormap_i_l[i], customLimits=None)


            self.m_concretePlotFunction(self.fig.axes[3],analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",0, color=colormap_i_r[i], customLimits=None,legendLabel=legend)

            self.m_concretePlotFunction(self.fig.axes[4],analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",1, color=colormap_i_r[i], customLimits=None)

            self.m_concretePlotFunction(self.fig.axes[5],analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",2, color=colormap_i_r[i], customLimits=None)
            i+=1

        



    def plotPanel(self):
        """
        Generates and plots the kinematic comparison panel.

        This method orchestrates the plotting process, including setting up the layers,
        arranging data, and rendering the final plot.

        Returns:
            matplotlib.figure.Figure: The generated plot as a matplotlib figure object.
            
        Raises:
            Exception: If the concrete plot function is not defined.
        """

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()
        self.fig.axes[0].legend(fontsize=6)
        self.fig.axes[3].legend(fontsize=6)
        # self.__setLegend(0)