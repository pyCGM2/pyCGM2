# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--


import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import pyCGM2
from pyCGM2.Report import plot

LOGGER = pyCGM2.LOGGER

# --- abstract procedure
class ClassificationProcedure(object):
    def __init__(self):
        pass


class PFKEprocedure(ClassificationProcedure):
    """PlantarFlexor-KneeExtensor classification procedure defined by Sangeux et al 2015

    Args:
        normativeData (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance
        midStanceDefinition (str): mid stance frame boundaries. Choice: PFKE (20,45), Perry (10,30) or Baker (10,50))
        dataType (str): chosen data type for classification. Choice cycle,mean,median.If cycle, knee and ankle scores are the median score from all cycle values with the normative mean value
        side (str): event context (Both,Left or Right)

    **Reference**

      - Sangeux, Morgan; Rodda, Jill; Graham, H. Kerr (2015)
      Sagittal gait patterns in cerebral palsy: the plantarflexor-knee extension couple index.
      In : Gait & posture, vol. 41, n° 2, p. 586–591. DOI: 10.1016/j.gaitpost.2014.12.019.



    """

    def __init__(self, normativeData,midStanceDefinition = "PFKE",dataType = "cycle",side = "Both"):
        super(PFKEprocedure, self).__init__()

        self.m_normativeData = normativeData

        if midStanceDefinition == "PFKE":
            self.m_frameLower = 20
            self.m_frameUpper = 45
        elif midStanceDefinition =="Perry":
            self.m_frameLower = 10
            self.m_frameUpper = 30
        elif midStanceDefinition =="Baker":
            self.m_frameLower = 10
            self.m_frameUpper = 50

        if  dataType not in ["cycle","mean","median"]:
            raise Exception("uncorrect dataType (choice : cycle, mean, median)")

        self.m_dataType = dataType

        if  side not in ["Both","Left","Right"]:
            raise Exception("uncorrect side (choice : Both, Left, Right)")

        self.m_side = side


    def __classify(self,ankle,knee):
        if ankle > -1 and ankle < 1 and knee > -1 and knee < 1:
            SagPatternClass = 'WNL'
        elif ankle < -1:
            if knee < 1:
                SagPatternClass = 'True equinus'
            elif knee > 1:
                SagPatternClass = 'Jump'
        elif ankle > 1:
            if knee > 1:
                SagPatternClass = 'Crouch'
            elif knee > -1:
                SagPatternClass = 'Undetermined 1'
            else:
                SagPatternClass = 'Impossible?'
        elif knee > 1:
                SagPatternClass = 'Apparent equinus'
        elif knee < -1:
            SagPatternClass = 'Knee recurvatum'
        else:
            SagPatternClass = 'Weird'

        PFKE = np.array([ankle, ankle, knee, knee])
        Boundaries = np.array([1,-1,1,-1])
        if SagPatternClass == "True equinus":
            DistPFKE = min(abs(PFKE[0:3]-Boundaries[0:3]))
        else:
            DistPFKE = min(abs(PFKE-Boundaries))

        return SagPatternClass,DistPFKE

    def run(self, analysis,pointSuffix):
        """run the procedure

        Args:
            analysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
            pointSuffix (str): suffix added to model outputs.

        """

        if pointSuffix is None:
            pointSuffix =""
        else:
            pointSuffix = "_"+pointSuffix


        self.m_sagPattern = dict()
        self.m_sagPattern["Left"] = dict()
        self.m_sagPattern["Right"] = dict()


        normalKnee = self.m_normativeData.data["KneeAngles"]
        normalAnkle = self.m_normativeData.data["AnkleAngles"]


        lower = self.m_frameLower
        upper = self.m_frameUpper

        if normalKnee["mean"].shape[0] == 51:
            lower_normal = int(self.m_frameLower/2)
            upper_normal = int((self.m_frameUpper+1)/2)
            step = 2
        else:
            lower_normal = lower
            upper_normal = upper
            step = 1

        if self.m_side in ["Both","Left"]:

            LKnee = analysis.kinematicStats.data["LKneeAngles"+pointSuffix, "Left"]
            LAnkle = analysis.kinematicStats.data["LAnkleAngles"+pointSuffix, "Left"]

            if self.m_dataType == "cycle":
                LKnee_score = list()
                LAnkle_score = list()
                for i in range(0,len(LKnee["values"])):
                    LKnee_score.append( np.mean((LKnee["values"][i][lower:upper:step, 0]
                                        - normalKnee["mean"][lower_normal:upper_normal, 0]) / normalKnee["sd"][lower_normal:upper_normal, 0]))
                    LAnkle_score.append( np.mean((LAnkle["values"][i][lower:upper:step, 0]
                                        - normalAnkle["mean"][lower_normal:upper_normal, 0]) / normalAnkle["sd"][lower_normal:upper_normal, 0]))

                LKnee_score = np.median(LKnee_score)
                LAnkle_score = np.median(LAnkle_score)
            else:

                LKnee_score = np.mean((LKnee[self.m_dataType][lower:upper:step, 0] - normalKnee["mean"]
                                             [lower_normal:upper_normal, 0]) / normalKnee["sd"][lower_normal:upper_normal, 0])
                LAnkle_score = np.mean((LAnkle[self.m_dataType][lower:upper:step, 0] - normalAnkle["mean"]
                                              [lower_normal:upper_normal, 0]) / normalAnkle["sd"][lower_normal:upper_normal, 0])


            sagClass,d_pfke = self.__classify(LAnkle_score,LKnee_score)
            self.m_sagPattern["Left"]["Class"] = sagClass
            self.m_sagPattern["Left"]["d"] = d_pfke
            self.m_sagPattern["Left"]["KneePFKE"] = LKnee_score
            self.m_sagPattern["Left"]["AnklePFKE"] = LAnkle_score

        if self.m_side in ["Both","Right"]:

            RKnee = analysis.kinematicStats.data["RKneeAngles"+pointSuffix, "Right"]
            RAnkle = analysis.kinematicStats.data["RAnkleAngles"+pointSuffix, "Right"]

            if self.m_dataType == "cycle":

                RKnee_score = list()
                RAnkle_score = list()
                for i in range(0,len(RKnee["values"])):
                    RKnee_score.append( np.mean((RKnee["values"][i][lower:upper:step, 0]
                                        - normalKnee["mean"][lower_normal:upper_normal, 0]) / normalKnee["sd"][lower_normal:upper_normal, 0]))
                    RAnkle_score.append( np.mean((RAnkle["values"][i][lower:upper:step, 0]
                                        - normalAnkle["mean"][lower_normal:upper_normal, 0]) / normalAnkle["sd"][lower_normal:upper_normal, 0]))

                RKnee_score = np.median(RKnee_score)
                RAnkle_score = np.median(RAnkle_score)

            else:

                RKnee_score = np.mean((RKnee[self.m_dataType][lower:upper:step, 0] - normalKnee["mean"]
                                             [lower_normal:upper_normal, 0]) / normalKnee["sd"][lower_normal:upper_normal, 0])
                RAnkle_score = np.mean((RAnkle[self.m_dataType][lower:upper:step, 0] - normalAnkle["mean"]
                                              [lower_normal:upper_normal, 0]) / normalAnkle["sd"][lower_normal:upper_normal, 0])


            sagClass,d_pfke = self.__classify(RAnkle_score,RKnee_score)
            self.m_sagPattern["Right"]["Class"] = sagClass
            self.m_sagPattern["Right"]["d"] = d_pfke
            self.m_sagPattern["Right"]["KneePFKE"] = RKnee_score
            self.m_sagPattern["Right"]["AnklePFKE"] = RAnkle_score

        return self.m_sagPattern

    def plot(self,analysis, title=None):
        """plot PFKE panels

        Args:
            analysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        """
        fig = plt.figure(facecolor="white")
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 3])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        title = "PFKE" + title if title is not None else "PFKE"

        fig.suptitle(title)

        ax0 =  fig.add_subplot(gs[0, 0]) #plt.subplot(2,2,1)
        ax1 =  fig.add_subplot(gs[0, 1]) #plt.subplot(2,2,3)
        ax2 =  fig.add_subplot(gs[1, 0:]) #plt.subplot(2,2,2)

        plot.descriptivePlot(ax0,analysis.kinematicStats,
                        "LKneeAngles","Left",0,
                        color="red",
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None)
        plot.descriptivePlot(ax0,analysis.kinematicStats,
                        "RKneeAngles","Right",0,
                        color="blue",
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None)

        fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData.data["KneeAngles"]["mean"].shape[0]),
            self.m_normativeData.data["KneeAngles"]["mean"][:,0]-self.m_normativeData.data["KneeAngles"]["sd"][:,0],
            self.m_normativeData.data["KneeAngles"]["mean"][:,0]+self.m_normativeData.data["KneeAngles"]["sd"][:,0],
            facecolor="green", alpha=0.5,linewidth=0)


        plot.descriptivePlot(ax1,analysis.kinematicStats,
                        "LAnkleAngles","Left",0,
                        color="red",
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None)
        plot.descriptivePlot(ax1,analysis.kinematicStats,
                        "RAnkleAngles","Right",0,
                        color="blue",
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None)

        fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData.data["AnkleAngles"]["mean"].shape[0]),
            self.m_normativeData.data["AnkleAngles"]["mean"][:,0]-self.m_normativeData.data["AnkleAngles"]["sd"][:,0],
            self.m_normativeData.data["AnkleAngles"]["mean"][:,0]+self.m_normativeData.data["AnkleAngles"]["sd"][:,0],
            facecolor="green", alpha=0.5,linewidth=0)

        from matplotlib.patches import Rectangle

        ax2.plot(self.m_sagPattern["Right"]["AnklePFKE"],self.m_sagPattern["Right"]["KneePFKE"],"*b")
        ax2.plot(self.m_sagPattern["Left"]["AnklePFKE"],self.m_sagPattern["Left"]["KneePFKE"],"*r")

        ax2.set_xlim([-6,6])
        ax2.set_ylim([-6,6])
        ax2.axvline(-1,color="black",ls='dashed')
        ax2.axvline(1,color="black",ls='dashed')
        ax2.axhline(1,color="black",ls='dashed')
        # ax2.axhline(-1,color="black",ls='dashed')

        ax2.spines["left"].set_position(("data", 0))
        ax2.spines["right"].set_position(("data", 0))
        ax2.spines["bottom"].set_position(("data", 0))
        ax2.spines["top"].set_position(("data", 0))
        ax2.plot(1, 0, ">k", transform=ax2.get_yaxis_transform(), clip_on=False)
        ax2.plot([-1,6],[-1,-1],ls="solid", color="black")

        ax2.add_patch(Rectangle((1, 1), 5, 5,
                     facecolor = 'green',alpha=0.5,
                     fill=True))
        ax2.add_patch(Rectangle((-1, 1), 2, 5,
                     facecolor = 'purple',alpha=0.5,
                     fill=True))
        ax2.add_patch(Rectangle((-6, 0), 5, 6,
                     facecolor = 'orange',alpha=0.5,
                     fill=True))
        ax2.add_patch(Rectangle((-6, -6), 5, 6,
                     facecolor = 'red',alpha=0.5,
                     fill=True))


        ax2.text(-5, 3, "Jump", fontsize=6)
        ax2.text(-5, -3, "True \n equinus", fontsize=6)
        ax2.text(5, 3, "Crouch", fontsize=6)
        ax2.text(0, 3, "Apparent \n equinus", fontsize=6,rotation=90)

        ax0.set_title("Knee Flexion" ,size=8)
        ax1.set_title("Ankle Flexion" ,size=8)

        ax2.set_title("PFKE Score" ,size=8)
        return fig
