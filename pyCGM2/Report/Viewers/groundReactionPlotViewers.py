import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

import matplotlib.pyplot as plt
import scipy  as sp

# pyCGM2
import pyCGM2
from pyCGM2.Report.Viewers import plotViewers


class NormalizedGroundReactionForcePlotViewer(plotViewers.AbstractPlotViewer):
    """ Plot time-Normalized Ground reaction forces 

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
    """

    def __init__(self,iAnalysis,pointLabelSuffix=None):

        super(NormalizedGroundReactionForcePlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

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





        if not self.m_automaticYlim_flag:
            #ax0.set_ylim([-2.0 *1000.0, 3.0*1000.0])
            pass
       
        for ax in self.fig.axes[0:3]:    
            ax.set_xlabel("Cycle %",size=8)
            ax.set_xlabel("Cycle %",size=8)
            ax.set_xlabel("Cycle %",size=8)


    def setNormativeDataset(self,iNormativeDataSet):
        """ Set the normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): a normative dataset instance
        """

        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        self.m_analysis.kineticStats.data["LStanGroundReactionForce"+suffixPlus,"Left"]["mean"][:,0]

        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",0, color="red", customLimits=None)
        

        self.m_concretePlotFunction(self.fig.axes[0],self.m_analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",0, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",1, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[1],self.m_analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",1, color="blue", customLimits=None)

        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kineticStats,
                "LStanGroundReactionForce"+suffixPlus,"Left",2, color="red", customLimits=None)
        self.m_concretePlotFunction(self.fig.axes[2],self.m_analysis.kineticStats,
                "RStanGroundReactionForce"+suffixPlus,"Right",2, color="blue", customLimits=None)
        


    def plotPanel(self):
        """Plot the panel"""

        if self.m_concretePlotFunction is None:
            raise Exception ("[pyCGM2] need definition of the concrete plot function")

        self.__setLayer()
        self.__setData()

        return self.fig
    

        # if self.m_normativeData is not None:
        #     if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
        #         self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
        #             self.m_normativeData["HipMoment"]["mean"][:,0]-self.m_normativeData["HipMoment"]["sd"][:,0],
        #             self.m_normativeData["HipMoment"]["mean"][:,0]+self.m_normativeData["HipMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
        #             self.m_normativeData["HipMoment"]["mean"][:,1]-self.m_normativeData["HipMoment"]["sd"][:,1],
        #             self.m_normativeData["HipMoment"]["mean"][:,1]+self.m_normativeData["HipMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["HipPower"]["mean"].shape[0]),
        #             self.m_normativeData["HipPower"]["mean"][:,2]-self.m_normativeData["HipPower"]["sd"][:,2],
        #             self.m_normativeData["HipPower"]["mean"][:,2]+self.m_normativeData["HipPower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)


        #         self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
        #             self.m_normativeData["KneeMoment"]["mean"][:,0]-self.m_normativeData["KneeMoment"]["sd"][:,0],
        #             self.m_normativeData["KneeMoment"]["mean"][:,0]+self.m_normativeData["KneeMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
        #             self.m_normativeData["KneeMoment"]["mean"][:,1]-self.m_normativeData["KneeMoment"]["sd"][:,1],
        #             self.m_normativeData["KneeMoment"]["mean"][:,1]+self.m_normativeData["KneeMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["KneePower"]["mean"].shape[0]),
        #             self.m_normativeData["KneePower"]["mean"][:,2]-self.m_normativeData["KneePower"]["sd"][:,2],
        #             self.m_normativeData["KneePower"]["mean"][:,2]+self.m_normativeData["KneePower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)



        #         self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
        #             self.m_normativeData["AnkleMoment"]["mean"][:,0]-self.m_normativeData["AnkleMoment"]["sd"][:,0],
        #             self.m_normativeData["AnkleMoment"]["mean"][:,0]+self.m_normativeData["AnkleMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
        #             self.m_normativeData["AnkleMoment"]["mean"][:,1]-self.m_normativeData["AnkleMoment"]["sd"][:,1],
        #             self.m_normativeData["AnkleMoment"]["mean"][:,1]+self.m_normativeData["AnkleMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)


        #         self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["AnklePower"]["mean"].shape[0]),
        #             self.m_normativeData["AnklePower"]["mean"][:,2]-self.m_normativeData["AnklePower"]["sd"][:,2],
        #             self.m_normativeData["AnklePower"]["mean"][:,2]+self.m_normativeData["AnklePower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)

        # return self.fig


class NormalizedGaitGrfIntegrationPlotViewer(plotViewers.AbstractPlotViewer):
    """ Plot the time-normalized Com Kinematics from integration of the mean ground reaction force 

    Args:
        iAnalysis (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
    """

    def __init__(self,iAnalysis,pointLabelSuffix=None,bodymass = 0):
               

        super(NormalizedGaitGrfIntegrationPlotViewer, self).__init__(iAnalysis)

        self.m_analysis = self.m_input
        if isinstance(self.m_analysis,pyCGM2.Processing.analysis.Analysis):
            pass
        else:
            LOGGER.logger.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_normativeData = None
        self.m_bodymass = bodymass





    def __setLayer(self):

        # stance phases.
        stanceL = self.m_analysis.kineticStats.pst['stancePhase', "Left"]["mean"]
        double1L = self.m_analysis.kineticStats.pst['doubleStance1', "Left"]["mean"]
        double2L = self.m_analysis.kineticStats.pst['doubleStance2', "Left"]["mean"]
        stanceR = self.m_analysis.kineticStats.pst['stancePhase', "Right"]["mean"]
        double1R = self.m_analysis.kineticStats.pst['doubleStance1', "Right"]["mean"]
        double2R = self.m_analysis.kineticStats.pst['doubleStance2', "Right"]["mean"]            




        self.fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Time-normalized Ground reaction force Integration from the mean traces\n """

        self.fig.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        nrow = 4 

        ax0 = plt.subplot(nrow,3,1)# 
        ax1 = plt.subplot(nrow,3,2)# 
        ax2 = plt.subplot(nrow,3,3)# 

        ax3 = plt.subplot(nrow,3,4)
        ax4 = plt.subplot(nrow,3,5)
        ax5 = plt.subplot(nrow,3,6)
        ax6 = plt.subplot(nrow,3,7)
        ax7 = plt.subplot(nrow,3,8)
        ax8 = plt.subplot(nrow,3,9)
        ax9 = plt.subplot(nrow,3,10)
        ax10 = plt.subplot(nrow,3,11)
        ax11 = plt.subplot(nrow,3,12)


        # - titles -
        self.fig.axes[0].set_title("Total Longitudinal Force" ,size=8)
        self.fig.axes[1].set_title("Total Lateral Force" ,size=8)
        self.fig.axes[2].set_title("Total Vertical Force" ,size=8)


        self.fig.axes[3].set_title("Longitudinal COM acceleration" ,size=8)
        self.fig.axes[4].set_title("Lateral COM acceleration" ,size=8) 
        self.fig.axes[5].set_title("Vertical COM acceleration" ,size=8)    
        
        self.fig.axes[6].set_title("Longitudinal COM velocity " ,size=8)
        self.fig.axes[7].set_title("Lateral COM velocity " ,size=8) 
        self.fig.axes[8].set_title("Vertical COM velocity " ,size=8)    
        self.fig.axes[9].set_title("Longitudinal COM position " ,size=8)
        self.fig.axes[10].set_title("Lateral COM position " ,size=8) 
        self.fig.axes[11].set_title("Vertical COM position " ,size=8)    

        # - lines -
        i=0
        for ax in self.fig.axes:
            color = "blue"
            ax.axvline(stanceL,color=color,ls='dashed')
            ax.axvline(double1L,ymin=0.9, ymax=1.0,color=color,ls='dotted')
            ax.axvline(stanceL-double2L,ymin=0.9, ymax=1.0,color=color,ls='dotted')

            color = "red"
            ax.axvline(stanceR,color=color,ls='dashed')
            ax.axvline(double1R,ymin=0.9, ymax=1.0,color=color,ls='dotted')
            ax.axvline(stanceR-double2R,ymin=0.9, ymax=1.0,color=color,ls='dotted')

            if i==2:
                if self.m_bodymass !=0:  ax.axhline(self.m_bodymass*9.81,color="black",ls='dashed')  
            else:
                ax.axhline(0,color="black",ls='dashed')  

            i+=1


        # limits

        accelerationL= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"])]

        accelerationR= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"])]
        
        velocityL= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"])]

        velocityR= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"])]
        
        positionL= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"])]

        positionR= [np.nanmin( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"]),
                         np.nanmax( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"])]

        if not self.m_automaticYlim_flag:
            for i in range(3,6):    
                self.fig.axes[i].set_ylim([min(accelerationL[0],accelerationR[0]) , max(accelerationL[1],accelerationR[1])])
            for i in range(6,9):    
                self.fig.axes[i].set_ylim([min(velocityL[0],velocityR[0]) , max(velocityL[1],velocityR[1])])
            for i in range(9,12):    
                self.fig.axes[i].set_ylim([min(positionL[0],positionR[0]) , max(positionL[1],positionR[1])])
            #ax0.set_ylim([-2.0 *1000.0, 3.0*1000.0])
            pass
       

        # - ticks - 
        for ax in self.fig.axes:
            ax.tick_params(axis='x', which='major', labelsize=6)
            ax.tick_params(axis='y', which='major', labelsize=6)


        # labels
        self.fig.axes[0].set_ylabel("Force [N]")
        self.fig.axes[3].set_ylabel("Acceleration [m.s-2]")
        self.fig.axes[6].set_ylabel("Velocity [m.s-1]")
        self.fig.axes[9].set_ylabel("Position [m.]")              

        for ax in self.fig.axes[9:12]:    
            ax.set_xlabel("stance %",size=8)
            ax.set_xlabel("stance %",size=8)
            ax.set_xlabel("stance %",size=8)


    def setNormativeDataset(self,iNormativeDataSet):
        """ Set the normative dataset

        Args:
            iNormativeDataSet (pyCGM2.Report.normativeDatasets.NormativeData): a normative dataset instance
        """

        self.m_normativeData = iNormativeDataSet.data

    def __setData(self):
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix is not None else ""

        self.fig.axes[0].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][:,0],"-r")
        self.fig.axes[1].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][:,1],"-r")
        self.fig.axes[2].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][:,2],"-r")
        self.fig.axes[3].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,0],"-r")
        self.fig.axes[4].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,1],"-r")
        self.fig.axes[5].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,2],"-r")
            
        self.fig.axes[6].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,0],"-r")
        self.fig.axes[7].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,1],"-r")
        self.fig.axes[8].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,2],"-r")
        self.fig.axes[9].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"][:,0],"-r")
        self.fig.axes[10].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"][:,1],"-r")
        self.fig.axes[11].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"][:,2],"-r")

        self.fig.axes[0].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][:,0],"-b")
        self.fig.axes[1].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][:,1],"-b")
        self.fig.axes[2].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][:,2],"-b")
        self.fig.axes[3].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,0],"-b")
        self.fig.axes[4].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,1],"-b")
        self.fig.axes[5].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,2],"-b")
        self.fig.axes[6].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,0],"-b")
        self.fig.axes[7].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,1],"-b")
        self.fig.axes[8].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,2],"-b")
        self.fig.axes[9].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"][:,0],"-b")
        self.fig.axes[10].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"][:,1],"-b")
        self.fig.axes[11].plot( self.m_analysis.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"][:,2],"-b")


    def plotPanel(self):
        """Plot the panel"""


        self.__setLayer()
        self.__setData()




        return self.fig
    

        # if self.m_normativeData is not None:
        #     if self.m_bodyPart == enums.BodyPartPlot.LowerLimb:
        #         self.fig.axes[0].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
        #             self.m_normativeData["HipMoment"]["mean"][:,0]-self.m_normativeData["HipMoment"]["sd"][:,0],
        #             self.m_normativeData["HipMoment"]["mean"][:,0]+self.m_normativeData["HipMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[1].fill_between(np.linspace(0,100,self.m_normativeData["HipMoment"]["mean"].shape[0]),
        #             self.m_normativeData["HipMoment"]["mean"][:,1]-self.m_normativeData["HipMoment"]["sd"][:,1],
        #             self.m_normativeData["HipMoment"]["mean"][:,1]+self.m_normativeData["HipMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[3].fill_between(np.linspace(0,100,self.m_normativeData["HipPower"]["mean"].shape[0]),
        #             self.m_normativeData["HipPower"]["mean"][:,2]-self.m_normativeData["HipPower"]["sd"][:,2],
        #             self.m_normativeData["HipPower"]["mean"][:,2]+self.m_normativeData["HipPower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)


        #         self.fig.axes[4].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
        #             self.m_normativeData["KneeMoment"]["mean"][:,0]-self.m_normativeData["KneeMoment"]["sd"][:,0],
        #             self.m_normativeData["KneeMoment"]["mean"][:,0]+self.m_normativeData["KneeMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[5].fill_between(np.linspace(0,100,self.m_normativeData["KneeMoment"]["mean"].shape[0]),
        #             self.m_normativeData["KneeMoment"]["mean"][:,1]-self.m_normativeData["KneeMoment"]["sd"][:,1],
        #             self.m_normativeData["KneeMoment"]["mean"][:,1]+self.m_normativeData["KneeMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[7].fill_between(np.linspace(0,100,self.m_normativeData["KneePower"]["mean"].shape[0]),
        #             self.m_normativeData["KneePower"]["mean"][:,2]-self.m_normativeData["KneePower"]["sd"][:,2],
        #             self.m_normativeData["KneePower"]["mean"][:,2]+self.m_normativeData["KneePower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)



        #         self.fig.axes[8].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
        #             self.m_normativeData["AnkleMoment"]["mean"][:,0]-self.m_normativeData["AnkleMoment"]["sd"][:,0],
        #             self.m_normativeData["AnkleMoment"]["mean"][:,0]+self.m_normativeData["AnkleMoment"]["sd"][:,0],
        #             facecolor="green", alpha=0.5,linewidth=0)

        #         self.fig.axes[9].fill_between(np.linspace(0,100,self.m_normativeData["AnkleMoment"]["mean"].shape[0]),
        #             self.m_normativeData["AnkleMoment"]["mean"][:,1]-self.m_normativeData["AnkleMoment"]["sd"][:,1],
        #             self.m_normativeData["AnkleMoment"]["mean"][:,1]+self.m_normativeData["AnkleMoment"]["sd"][:,1],
        #             facecolor="green", alpha=0.5,linewidth=0)


        #         self.fig.axes[11].fill_between(np.linspace(0,100,self.m_normativeData["AnklePower"]["mean"].shape[0]),
        #             self.m_normativeData["AnklePower"]["mean"][:,2]-self.m_normativeData["AnklePower"]["sd"][:,2],
        #             self.m_normativeData["AnklePower"]["mean"][:,2]+self.m_normativeData["AnklePower"]["sd"][:,2],
        #             facecolor="green", alpha=0.5,linewidth=0)

        # return self.fig