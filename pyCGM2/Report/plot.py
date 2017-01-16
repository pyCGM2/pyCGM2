# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

import pdb

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# pyCGM2
#import pyCGM2
import pyCGM2.Processing.analysis as CGM2analysis


# openMA
import ma.io
import ma.body



# ---- convenient plot - single Model------
def gaitKinematicsTemporalPlotPanel(trial,filename,pointLabelSuffix="",path = ""):
    """
        **Description :** convenient function for plotting kinematic gait trace from an openma-trial. 
        
        
        .. warning:: Point label must match vicon point nomenclature (ex: LHipAngles,LHipMoment,...)


        :Parameters:
             - `trial` (openma-trial) - openma trial from a c3d
             - `filename` (str) - c3d filename of the gait trial
             - `pointLabelSuffix` (str) - suffix ending conventional kinematic CGM labels
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - a pdf file name with extension "- Temporal Kinematics.pdf"
            - `pdfName` (str)  - filename of the output pdf


        **Usage**



        """

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

    firstFrame = trial.property("TRIAL:ACTUAL_START_FIELD").cast()[0]
    lastFrame = trial.property("TRIAL:ACTUAL_END_FIELD").cast()[0]

    end = lastFrame-firstFrame

    # --- left Kinematics ------
    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title= filename[:-4] + " - Temporal Kinematics "
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


    ax1 = plt.subplot(5,3,1)# Pelvis X
    ax2 = plt.subplot(5,3,2)# Pelvis Y
    ax3 = plt.subplot(5,3,3)# Pelvis Z
    ax4 = plt.subplot(5,3,4)# Hip X
    ax5 = plt.subplot(5,3,5)# Hip Y
    ax6 = plt.subplot(5,3,6)# Hip Z
    ax7 = plt.subplot(5,3,7)# Knee X
    ax8 = plt.subplot(5,3,8)# Knee Y
    ax9 = plt.subplot(5,3,9)# Knee Z
    ax10 = plt.subplot(5,3,10)# Ankle X
    ax11 = plt.subplot(5,3,11)# Ankle Y
    ax12 = plt.subplot(5,3,12)# Ankle Z
    ax13 = plt.subplot(5,3,15)# foot Progression Z

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13]

    ax1.set_ylim([0.0,60])
    ax2.set_ylim([-30,30])
    ax3.set_ylim([-30,30])
    ax4.set_ylim([-20,70])
    ax5.set_ylim([-30,30])
    ax6.set_ylim([-30,30])
    ax7.set_ylim([-15,75])
    ax8.set_ylim([-30,30])
    ax9.set_ylim([-30,30])
    ax10.set_ylim([-30,30])
    ax11.set_ylim([-30,30])
    ax12.set_ylim([-30,30])
    ax13.set_ylim([-30,30])

    ax1.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax2.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax3.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax4.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax5.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax6.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax7.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax8.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax9.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax10.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax11.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax12.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax13.plot(trial.findChild(ma.T_TimeSequence,"LFootProgressAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")


    ax1.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax2.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax3.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax4.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax5.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax6.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax7.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax8.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax9.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax10.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax11.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax12.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax13.plot(trial.findChild(ma.T_TimeSequence,"RFootProgressAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")


    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.set_title("Pelvis Tilt" ,size=8)
    ax2.set_title("Pelvis Obliquity" ,size=8)
    ax3.set_title("Pelvis Rotation" ,size=8)
    ax4.set_title("Hip Flexion" ,size=8)
    ax5.set_title("Hip Adduction" ,size=8)
    ax6.set_title("Hip Rotation" ,size=8)
    ax7.set_title("Knee Flexion" ,size=8)
    ax8.set_title("Knee Adduction" ,size=8)
    ax9.set_title("Knee Rotation" ,size=8)
    ax10.set_title("Ankle Flexion" ,size=8)
    ax11.set_title("Ankle Adduction" ,size=8)
    ax12.set_title("Ankle Rotation" ,size=8)
    ax13.set_title("Foot Progress" ,size=8)

    pdfName = filename[:-4] +"- Temporal Kinematics.pdf"
    pp = PdfPages(str(path+ pdfName))
    pp.savefig(fig)
    pp.close()

    return pdfName



def gaitKineticsTemporalPlotPanel(trial,filename,pointLabelSuffix="",path = ""):
    """
        **Description :** convenient function for plotting kinetic gait trace from an openma-trial

        .. warning:: Point label must match vicon CGM1 point label

        :Parameters:
             - `trial` (openma-trial) - openma trial
             - `filename` (str) - c3d filename of the gait trial
             - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - a pdf file name with extension "- Temporal Kinetics.pdf"
            - `pdfName` (str)  - filename of the output pdf


    """
    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

    firstFrame = trial.property("TRIAL:ACTUAL_START_FIELD").cast()[0]
    lastFrame = trial.property("TRIAL:ACTUAL_END_FIELD").cast()[0]

    end = lastFrame-firstFrame

    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title= filename[:-4] + " - Temporal Kinetics "
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(3,4,1)# Hip X extensor
    ax2 = plt.subplot(3,4,2)# Hip Y abductor
    ax3 = plt.subplot(3,4,3)# Hip Z rotation
    ax4 = plt.subplot(3,4,4)# Knee Z power

    ax5 = plt.subplot(3,4,5)# knee X extensor
    ax6 = plt.subplot(3,4,6)# knee Y abductor
    ax7 = plt.subplot(3,4,7)# knee Z rotation
    ax8 = plt.subplot(3,4,8)# knee Z power

    ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax10 = plt.subplot(3,4,10)# ankle Y rotation
    ax11 = plt.subplot(3,4,11)# ankle Z everter
    ax12 = plt.subplot(3,4,12)# ankle Z power

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

    ax1.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax2.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax3.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax4.plot(trial.findChild(ma.T_TimeSequence,"LHipPower"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")

    ax5.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax6.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax7.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax8.plot(trial.findChild(ma.T_TimeSequence,"LKneePower"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")

    ax9.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "r")
    ax10.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "r")
    ax11.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")
    ax12.plot(trial.findChild(ma.T_TimeSequence,"LAnklePower"+pointLabelSuffixPlus).data()[:,2], '-', color= "r")


    ax1.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax2.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax3.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax4.plot(trial.findChild(ma.T_TimeSequence,"RHipPower"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")

    ax5.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax6.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax7.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax8.plot(trial.findChild(ma.T_TimeSequence,"RKneePower"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")

    ax9.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= "b")
    ax10.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= "b")
    ax11.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")
    ax12.plot(trial.findChild(ma.T_TimeSequence,"RAnklePower"+pointLabelSuffixPlus).data()[:,2], '-', color= "b")

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.set_title("Hip extensor Moment" ,size=8)
    ax2.set_title("Hip adductor Moment" ,size=8)
    ax3.set_title("Hip rotation Moment" ,size=8)
    ax4.set_title("Hip power" ,size=8)

    ax5.set_title("Knee extensor Moment" ,size=8)
    ax6.set_title("Knee adductor Moment" ,size=8)
    ax7.set_title("Knee rotation Moment" ,size=8)
    ax8.set_title("Knee power" ,size=8)

    ax9.set_title("Ankle extensor Moment" ,size=8)
    ax10.set_title("Ankle adductor Moment" ,size=8)
    ax11.set_title("Ankle rotation Moment" ,size=8)
    ax12.set_title("Ankle power" ,size=8)


    pdfName = filename[:-4] +"- Temporal Kinetics.pdf"
    pp = PdfPages(str(path+ pdfName))
    pp.savefig(fig)
    pp.close()

    return pdfName

# ---- convenient plot - multiple Model------

# -- on trials --

def gaitKinematicsTemporal_multipleModel_PlotPanel(trials,labels,filename,pointLabelSuffix="",path = ""):


    #TODO : Is this function necessary ? 
    # 

    """
        **Description :** convenient function for plotting kinematic gait trace from a c3d processed with different models (a model = an openma::trial) .

        :Parameters:
             - `trials` (list ) - list of openma trials representing each one a process
             - `labels` (list)  - list of label matching trials
             - `filename` (str) - c3d filename of the gait trial
             - `pointLabelSuffix` (str) - suffix ending conventional CGM kinematic label
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - pdf filenames with extension "- multiModels - Left Temporal Kinematics.pdf"   and "- multiModels - Right Temporal Kinematics.pdf"

        **Usage**

        .. code:: python

            plot.gaitKinematicsTemporalPlotPanel([kinematicTrials_VICON, kinematicTrials_OPENMA, kinematicTrials_PYCGM2],
                                                 ["Vicon","openMA","pyCGM2"],
                                                 "gait 01.c3d")

    """

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix

    # input check
    n_trials = len(trials)

    if  not all(x.property("TRIAL:ACTUAL_START_FIELD").cast()[0] == trials[0].property("TRIAL:ACTUAL_START_FIELD").cast()[0] for x in trials):
        raise Exception("trial Instances don t have the same first frame.")

    if  not all(x.property("TRIAL:ACTUAL_END_FIELD").cast()[0] == trials[0].property("TRIAL:ACTUAL_END_FIELD").cast()[0] for x in trials):
        raise Exception("trial Instances don t have the same first frame.")



    colormap = plt.cm.gnuplot
    colormap_i=[colormap(i) for i in np.linspace(0, 1, len(trials))]


    firstFrame = trials[0].property("TRIAL:ACTUAL_START_FIELD").cast()[0]
    lastFrame = trials[0].property("TRIAL:ACTUAL_END_FIELD").cast()[0]

    end = lastFrame-firstFrame+1

    # --- left Kinematics ------
    fig_left = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=" LEFT Temporal Kinematics "
    fig_left.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


    ax1 = plt.subplot(5,3,1)# Pelvis X
    ax2 = plt.subplot(5,3,2)# Pelvis Y
    ax3 = plt.subplot(5,3,3)# Pelvis Z
    ax4 = plt.subplot(5,3,4)# Hip X
    ax5 = plt.subplot(5,3,5)# Hip Y
    ax6 = plt.subplot(5,3,6)# Hip Z
    ax7 = plt.subplot(5,3,7)# Knee X
    ax8 = plt.subplot(5,3,8)# Knee Y
    ax9 = plt.subplot(5,3,9)# Knee Z
    ax10 = plt.subplot(5,3,10)# Ankle X
    ax11 = plt.subplot(5,3,11)# Ankle Y
    ax12 = plt.subplot(5,3,12)# Ankle Z
    ax13 = plt.subplot(5,3,15)# foot Progression Z

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13]

    ax1.set_ylim([0.0,60])
    ax2.set_ylim([-30,30])
    ax3.set_ylim([-30,30])
    ax4.set_ylim([-20,70])
    ax5.set_ylim([-30,30])
    ax6.set_ylim([-30,30])
    ax7.set_ylim([-15,75])
    ax8.set_ylim([-30,30])
    ax9.set_ylim([-30,30])
    ax10.set_ylim([-30,30])
    ax11.set_ylim([-30,30])
    ax12.set_ylim([-30,30])
    ax13.set_ylim([-30,30])

    i=0
    for trial in trials:

        ax1.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax2.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax3.plot(trial.findChild(ma.T_TimeSequence,"LPelvisAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax4.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax5.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax6.plot(trial.findChild(ma.T_TimeSequence,"LHipAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax7.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax8.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax9.plot(trial.findChild(ma.T_TimeSequence,"LKneeAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax10.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax11.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax12.plot(trial.findChild(ma.T_TimeSequence,"LAnkleAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax13.plot(trial.findChild(ma.T_TimeSequence,"LFootProgressAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        i+=1

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 1., .102),
          ncol=len(trials), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Pelvis Tilt" ,size=8)
    ax2.set_title("Pelvis Obliquity" ,size=8)
    ax3.set_title("Pelvis Rotation" ,size=8)
    ax4.set_title("Hip Flexion" ,size=8)
    ax5.set_title("Hip Adduction" ,size=8)
    ax6.set_title("Hip Rotation" ,size=8)
    ax7.set_title("Knee Flexion" ,size=8)
    ax8.set_title("Knee Adduction" ,size=8)
    ax9.set_title("Knee Rotation" ,size=8)
    ax10.set_title("Ankle Flexion" ,size=8)
    ax11.set_title("Ankle Adduction" ,size=8)
    ax12.set_title("Ankle Rotation" ,size=8)
    ax13.set_title("Foot Progress" ,size=8)

    # --- right Kinematics ------
    fig_right = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=" RIGHT Temporal Kinematics "
    fig_right.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


    ax1 = plt.subplot(5,3,1)# Pelvis X
    ax2 = plt.subplot(5,3,2)# Pelvis Y
    ax3 = plt.subplot(5,3,3)# Pelvis Z
    ax4 = plt.subplot(5,3,4)# Hip X
    ax5 = plt.subplot(5,3,5)# Hip Y
    ax6 = plt.subplot(5,3,6)# Hip Z
    ax7 = plt.subplot(5,3,7)# Knee X
    ax8 = plt.subplot(5,3,8)# Knee Y
    ax9 = plt.subplot(5,3,9)# Knee Z
    ax10 = plt.subplot(5,3,10)# Ankle X
    ax11 = plt.subplot(5,3,11)# Ankle Y
    ax12 = plt.subplot(5,3,12)# Ankle Z
    ax13 = plt.subplot(5,3,15)# Ankle Z

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13]

    ax1.set_ylim([0.0,60])
    ax2.set_ylim([-30,30])
    ax3.set_ylim([-30,30])
    ax4.set_ylim([-20,70])
    ax5.set_ylim([-30,30])
    ax6.set_ylim([-30,30])
    ax7.set_ylim([-15,75])
    ax8.set_ylim([-30,30])
    ax9.set_ylim([-30,30])
    ax10.set_ylim([-30,30])
    ax11.set_ylim([-30,30])
    ax12.set_ylim([-30,30])
    ax13.set_ylim([-30,30])

    i=0
    for trial in trials:


        ax1.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax2.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax3.plot(trial.findChild(ma.T_TimeSequence,"RPelvisAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax4.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax5.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax6.plot(trial.findChild(ma.T_TimeSequence,"RHipAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax7.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax8.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax9.plot(trial.findChild(ma.T_TimeSequence,"RKneeAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax10.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax11.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax12.plot(trial.findChild(ma.T_TimeSequence,"RAnkleAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax13.plot(trial.findChild(ma.T_TimeSequence,"RFootProgressAngles"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        i+=1



    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 1., .102),
          ncol=len(trials), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Pelvis Tilt" ,size=8)
    ax2.set_title("Pelvis Obliquity" ,size=8)
    ax3.set_title("Pelvis Rotation" ,size=8)
    ax4.set_title("Hip Flexion" ,size=8)
    ax5.set_title("Hip Adduction" ,size=8)
    ax6.set_title("Hip Rotation" ,size=8)
    ax7.set_title("Knee Flexion" ,size=8)
    ax8.set_title("Knee Adduction" ,size=8)
    ax9.set_title("Knee Rotation" ,size=8)
    ax10.set_title("Ankle Flexion" ,size=8)
    ax11.set_title("Ankle Adduction" ,size=8)
    ax12.set_title("Ankle Rotation" ,size=8)
    ax13.set_title("Foot Progress" ,size=8)


    pp = PdfPages(str(path+ filename[:-4] +"-multiModels - Left Temporal Kinematics.pdf"))
    pp.savefig(fig_left)
    pp.close()

    pp = PdfPages(str(path+ filename[:-4] +"-multiModels - Right Temporal Kinematics.pdf"))
    pp.savefig(fig_right)
    pp.close()



def gaitKineticsTemporal_multipleModel_PlotPanel(trials,labels,filename,pointLabelSuffix,path = ""):

    #TODO : Is this function necessary ? 

    """

        **Description :** convenient function for plotting kinetic gait traces from a c3d processed with different models (a model = an openma::trial) .

        :Parameters:
             - `trials` (list ) - list of openma trials representing each one a process
             - `labels` (list)  - list of label matching trials
             - `filename` (str) - c3d filename of the gait trial
             - `pointLabelSuffix` (str) - suffix ending conventional CGM kinematic label
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - pdf filenames with extension "- multiModels - Left Temporal Kinetics.pdf"   and "- multiModels - Right Temporal Kinetics.pdf"

        **Usage**

        .. code:: python

            plot.gaitKineticsTemporalPlotPanel([kineticTrial_VICON, kineticTrial_OPENMA, kineticTrial_PYCGM2],
                                                     ["Vicon","openMA","pyCGM2"],
                                                     "gait 01.c3d")

    """

    pointLabelSuffixPlus  = pointLabelSuffix   if pointLabelSuffix =="" else "_"+pointLabelSuffix


    # input check
    n_trials = len(trials)
    if  not all(x.property("TRIAL:ACTUAL_START_FIELD").cast()[0] == trials[0].property("TRIAL:ACTUAL_START_FIELD").cast()[0] for x in trials):
        raise Exception("trial Instances don t have the same first frame.")

    if  not all(x.property("TRIAL:ACTUAL_END_FIELD").cast()[0] == trials[0].property("TRIAL:ACTUAL_END_FIELD").cast()[0] for x in trials):
        raise Exception("trial Instances don t have the same first frame.")


    colormap = plt.cm.gnuplot
    colormap_i=[colormap(i) for i in np.linspace(0, 1, len(trials))]

    firstFrame = trials[0].property("TRIAL:ACTUAL_START_FIELD").cast()[0]
    lastFrame = trials[0].property("TRIAL:ACTUAL_END_FIELD").cast()[0]

    end = lastFrame-firstFrame+1

    # --- left Kinetics ------
    fig_left = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=" LEFT Temporal Kinetics "
    fig_left.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(3,4,1)# Hip X extensor
    ax2 = plt.subplot(3,4,2)# Hip Y abductor
    ax3 = plt.subplot(3,4,3)# Hip Z rotation
    ax4 = plt.subplot(3,4,4)# Knee Z power

    ax5 = plt.subplot(3,4,5)# knee X extensor
    ax6 = plt.subplot(3,4,6)# knee Y abductor
    ax7 = plt.subplot(3,4,7)# knee Z rotation
    ax8 = plt.subplot(3,4,8)# knee Z power

    ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax10 = plt.subplot(3,4,10)# ankle Y rotation
    ax11 = plt.subplot(3,4,11)# ankle Z everter
    ax12 = plt.subplot(3,4,12)# ankle Z power

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

    i=0
    for trial in trials:


        ax1.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax2.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax3.plot(trial.findChild(ma.T_TimeSequence,"LHipMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax4.plot(trial.findChild(ma.T_TimeSequence,"LHipPower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        ax5.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax6.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax7.plot(trial.findChild(ma.T_TimeSequence,"LKneeMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax8.plot(trial.findChild(ma.T_TimeSequence,"LKneePower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        ax9.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax10.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax11.plot(trial.findChild(ma.T_TimeSequence,"LAnkleMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax12.plot(trial.findChild(ma.T_TimeSequence,"LAnklePower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])


        i+=1

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 1., .102),
          ncol=len(trials), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Hip extensor Moment" ,size=8)
    ax2.set_title("Hip adductor Moment" ,size=8)
    ax3.set_title("Hip rotation Moment" ,size=8)
    ax4.set_title("Hip power" ,size=8)

    ax5.set_title("Knee extensor Moment" ,size=8)
    ax6.set_title("Knee adductor Moment" ,size=8)
    ax7.set_title("Knee rotation Moment" ,size=8)
    ax8.set_title("Knee power" ,size=8)

    ax9.set_title("Ankle extensor Moment" ,size=8)
    ax10.set_title("Ankle adductor Moment" ,size=8)
    ax11.set_title("Ankle rotation Moment" ,size=8)
    ax12.set_title("Ankle power" ,size=8)

#    pp = PdfPages(str(path+ filename[:-4] +"-Left Temporal Kinetics.pdf"))
#    pp.savefig(fig_left)
#    pp.close()

     # --- right Kinetics ------
    fig_right = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=" Right Temporal Kinetics "
    fig_right.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(3,4,1)# Hip X extensor
    ax2 = plt.subplot(3,4,2)# Hip Y abductor
    ax3 = plt.subplot(3,4,3)# Hip Z rotation
    ax4 = plt.subplot(3,4,4)# Knee Z power

    ax5 = plt.subplot(3,4,5)# knee X extensor
    ax6 = plt.subplot(3,4,6)# knee Y abductor
    ax7 = plt.subplot(3,4,7)# knee Z rotation
    ax8 = plt.subplot(3,4,8)# knee Z power

    ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax10 = plt.subplot(3,4,10)# ankle Y rotation
    ax11 = plt.subplot(3,4,11)# ankle Z everter
    ax12 = plt.subplot(3,4,12)# ankle Z power

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

    i=0
    for trial in trials:


        ax1.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax2.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax3.plot(trial.findChild(ma.T_TimeSequence,"RHipMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax4.plot(trial.findChild(ma.T_TimeSequence,"RHipPower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        ax5.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax6.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax7.plot(trial.findChild(ma.T_TimeSequence,"RKneeMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax8.plot(trial.findChild(ma.T_TimeSequence,"RKneePower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        ax9.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,0], '-', color= colormap_i[i])
        ax10.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,1], '-', color= colormap_i[i])
        ax11.plot(trial.findChild(ma.T_TimeSequence,"RAnkleMoment"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])
        ax12.plot(trial.findChild(ma.T_TimeSequence,"RAnklePower"+pointLabelSuffixPlus).data()[:,2], '-', color= colormap_i[i])

        i+=1

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 1., .102),
          ncol=len(trials), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Hip extensor Moment" ,size=8)
    ax2.set_title("Hip adductor Moment" ,size=8)
    ax3.set_title("Hip rotation Moment" ,size=8)
    ax4.set_title("Hip power" ,size=8)

    ax5.set_title("Knee extensor Moment" ,size=8)
    ax6.set_title("Knee adductor Moment" ,size=8)
    ax7.set_title("Knee rotation Moment" ,size=8)
    ax8.set_title("Knee power" ,size=8)

    ax9.set_title("Ankle extensor Moment" ,size=8)
    ax10.set_title("Ankle adductor Moment" ,size=8)
    ax11.set_title("Ankle rotation Moment" ,size=8)
    ax12.set_title("Ankle power" ,size=8)

    pp = PdfPages(str(path+ filename[:-4] +"-multiModels - Left Temporal Kinetics.pdf"))
    pp.savefig(fig_left)
    pp.close()

    pp = PdfPages(str(path+ filename[:-4] +"-multiModels - Right Temporal Kinetics.pdf"))
    pp.savefig(fig_right)
    pp.close()


# -- on pyCGM2::cycles --

def gaitKinematicsCycleTemporal_multipleModel_PlotPanel(cycleInstances,labels,filename,path = ""):

    #TODO : Is this function necessary ? 

    '''
        .. todo:: IMPROVE

        **Description :** plot kinematic gait panel from pyCGM2.Processing.cycle.Cycles instances extracted from the same c3d with possible different process/Model

        :Parameters:
             - `cycleInstances` (list of pyCGM2.Processing.cycle.Cycles) - pyCGM2.Processing.cycle.Cycles instances
             - `labels` (list)  - list of label matching trials
             - `filename` (str) - c3d filename of the gait trial
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - matplotlib figure

        **Usage**

        .. code:: python

            #cycles_VICON is object built from a pyCGM2::cycleFilter

            plot.gaitKinematicsTemporalPlotPanel([cycles_VICON.kinematicCycles[1], cycles_OPENMA.kinematicCycles[1], cycles_PYCGM2.kinematicCycles[1]],
                                                     ["Vicon","openMA","pyCGM2"],
                                                     "gait 01.c3d")



   '''


    # input check
    n_cycleInstances = len(cycleInstances)
    if  not all(x.begin == cycleInstances[0].begin for x in cycleInstances):
        raise Exception("cycleInstances don t have the same first frame.")

    if  not  all(x.end == cycleInstances[0].end for x in cycleInstances):
        raise Exception("cycleInstances don t have the same last frame")

    colormap = plt.cm.gnuplot
    colormap_i=[colormap(i) for i in np.linspace(0, 1, len(cycleInstances))]


    firstFrame = cycleInstances[0].begin
    lastFrame = cycleInstances[0].end

    end = lastFrame-firstFrame+1


    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    if cycleInstances[0].context == "Left":
        title=" LEFT Cycle Temporal Kinematics"
    if cycleInstances[0].context == "Right":
        title=" RIGHT Cycle Temporal Kinematics"
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


    ax1 = plt.subplot(5,3,1)# Pelvis X
    ax2 = plt.subplot(5,3,2)# Pelvis Y
    ax3 = plt.subplot(5,3,3)# Pelvis Z
    ax4 = plt.subplot(5,3,4)# Hip X
    ax5 = plt.subplot(5,3,5)# Hip Y
    ax6 = plt.subplot(5,3,6)# Hip Z
    ax7 = plt.subplot(5,3,7)# Knee X
    ax8 = plt.subplot(5,3,8)# Knee Y
    ax9 = plt.subplot(5,3,9)# Knee Z
    ax10 = plt.subplot(5,3,10)# Ankle X
    ax11 = plt.subplot(5,3,11)# Ankle Y
    ax12 = plt.subplot(5,3,12)# Ankle Z
    ax13 = plt.subplot(5,3,15)# foot Progression Z

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13]

    i=0
    for cycleIt in cycleInstances:

        if cycleIt.context == "Left":
            ax1.plot(cycleIt.getPointTimeSequenceData("LPelvisAngles")[:,0], '-', color= colormap_i[i])
            ax2.plot(cycleIt.getPointTimeSequenceData("LPelvisAngles")[:,1], '-', color= colormap_i[i])
            ax3.plot(cycleIt.getPointTimeSequenceData("LPelvisAngles")[:,2], '-', color= colormap_i[i])
            ax4.plot(cycleIt.getPointTimeSequenceData("LHipAngles")[:,0], '-', color= colormap_i[i])
            ax5.plot(cycleIt.getPointTimeSequenceData("LHipAngles")[:,1], '-', color= colormap_i[i])
            ax6.plot(cycleIt.getPointTimeSequenceData("LHipAngles")[:,2], '-', color= colormap_i[i])
            ax7.plot(cycleIt.getPointTimeSequenceData("LKneeAngles")[:,0], '-', color= colormap_i[i])
            ax8.plot(cycleIt.getPointTimeSequenceData("LKneeAngles")[:,1], '-', color= colormap_i[i])
            ax9.plot(cycleIt.getPointTimeSequenceData("LKneeAngles")[:,2], '-', color= colormap_i[i])
            ax10.plot(cycleIt.getPointTimeSequenceData("LAnkleAngles")[:,0], '-', color= colormap_i[i])
            ax11.plot(cycleIt.getPointTimeSequenceData("LAnkleAngles")[:,1], '-', color= colormap_i[i])
            ax12.plot(cycleIt.getPointTimeSequenceData("LAnkleAngles")[:,2], '-', color= colormap_i[i])
            ax13.plot(cycleIt.getPointTimeSequenceData("LFootProgressAngles")[:,2], '-', color= colormap_i[i])

        if cycleIt.context == "Right":
            ax1.plot(cycleIt.getPointTimeSequenceData("RPelvisAngles")[:,0], '-', color= colormap_i[i])
            ax2.plot(cycleIt.getPointTimeSequenceData("RPelvisAngles")[:,1], '-', color= colormap_i[i])
            ax3.plot(cycleIt.getPointTimeSequenceData("RPelvisAngles")[:,2], '-', color= colormap_i[i])
            ax4.plot(cycleIt.getPointTimeSequenceData("RHipAngles")[:,0], '-', color= colormap_i[i])
            ax5.plot(cycleIt.getPointTimeSequenceData("RHipAngles")[:,1], '-', color= colormap_i[i])
            ax6.plot(cycleIt.getPointTimeSequenceData("RHipAngles")[:,2], '-', color= colormap_i[i])
            ax7.plot(cycleIt.getPointTimeSequenceData("RKneeAngles")[:,0], '-', color= colormap_i[i])
            ax8.plot(cycleIt.getPointTimeSequenceData("RKneeAngles")[:,1], '-', color= colormap_i[i])
            ax9.plot(cycleIt.getPointTimeSequenceData("RKneeAngles")[:,2], '-', color= colormap_i[i])
            ax10.plot(cycleIt.getPointTimeSequenceData("RAnkleAngles")[:,0], '-', color= colormap_i[i])
            ax11.plot(cycleIt.getPointTimeSequenceData("RAnkleAngles")[:,1], '-', color= colormap_i[i])
            ax12.plot(cycleIt.getPointTimeSequenceData("RAnkleAngles")[:,2], '-', color= colormap_i[i])
            ax13.plot(cycleIt.getPointTimeSequenceData("RFootProgressAngles")[:,2], '-', color= colormap_i[i])

        i+=1

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 1., .102),
          ncol=len(cycleInstances), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Pelvis Tilt" ,size=8)
    ax2.set_title("Pelvis Obliquity" ,size=8)
    ax3.set_title("Pelvis Rotation" ,size=8)
    ax4.set_title("Hip Flexion" ,size=8)
    ax5.set_title("Hip Adduction" ,size=8)
    ax6.set_title("Hip Rotation" ,size=8)
    ax7.set_title("Knee Flexion" ,size=8)
    ax8.set_title("Knee Adduction" ,size=8)
    ax9.set_title("Knee Rotation" ,size=8)
    ax10.set_title("Ankle Flexion" ,size=8)
    ax11.set_title("Ankle Adduction" ,size=8)
    ax12.set_title("Ankle Rotation" ,size=8)
    ax13.set_title("Foot Progress" ,size=8)


def gaitKineticsCycleTemporal_multipleModel_PlotPanel(cycleInstances,labels,filename="",path = ""):


    #TODO : Is this function necessary ? 

    '''

        .. todo:: IMPROVE

        **Description :** plot kinetic gait panel from pyCGM2.Processing.cycle.Cycles instances extracted from the same c3d with possible different process/Model

        :Parameters:
             - `cycleInstances` (list of pyCGM2.Processing.cycle.Cycles) - pyCGM2.Processing.cycle.Cycles instances
             - `labels` (list)  - list of label matching trials
             - `filename` (str) - c3d filename of the gait trial
             - `path` (str) - path pointing a folder where pdf will be stored. Must end with \\

        :Return:
            - matplotlib figure

        **Usage**

        .. code:: python

            #cycles_VICON is object built from a pyCGM2::cycleFilter

            plot.gaitKinematicsTemporalPlotPanel([cycles_VICON.kinematicCycles[1], cycles_OPENMA.kinematicCycles[1], cycles_PYCGM2.kinematicCycles[1]],
                                                     ["Vicon","openMA","pyCGM2"],
                                                     "gait 01.c3d")



   '''

    # input check
    n_cycleInstances = len(cycleInstances)
    if  not all(x.begin == cycleInstances[0].begin for x in cycleInstances):
        raise Exception("cycleInstances don t have the same first frame.")

    if  not all(x.end == cycleInstances[0].end for x in cycleInstances):
        raise Exception("cycleInstances don t have the same last frame")


    colormap = plt.cm.gnuplot
    colormap_i=[colormap(i) for i in np.linspace(0, 1, len(cycleInstances))]


    firstFrame = cycleInstances[0].begin
    lastFrame = cycleInstances[0].end

    end = lastFrame-firstFrame+1


    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    if cycleInstances[0].context == "Left":
        title=" LEFT Cycle Temporal Kinetics"
    if cycleInstances[0].context == "Right":
        title=" RIGHT Cycle Temporal Kinetics"
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(3,4,1)# Hip X extensor
    ax2 = plt.subplot(3,4,2)# Hip Y abductor
    ax3 = plt.subplot(3,4,3)# Hip Z rotation
    ax4 = plt.subplot(3,4,4)# Knee Z power

    ax5 = plt.subplot(3,4,5)# knee X extensor
    ax6 = plt.subplot(3,4,6)# knee Y abductor
    ax7 = plt.subplot(3,4,7)# knee Z rotation
    ax8 = plt.subplot(3,4,8)# knee Z power

    ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax10 = plt.subplot(3,4,10)# ankle Y rotation
    ax11 = plt.subplot(3,4,11)# ankle Z everter
    ax12 = plt.subplot(3,4,12)# ankle Z power

    axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

    i=0
    for cycleIt in cycleInstances:

        if cycleIt.context == "Left":
            ax1.plot(cycleIt.getPointTimeSequenceData("LHipMoment")[:,0], '-', color= colormap_i[i])
            ax2.plot(cycleIt.getPointTimeSequenceData("LHipMoment")[:,1], '-', color= colormap_i[i])
            ax3.plot(cycleIt.getPointTimeSequenceData("LHipMoment")[:,2], '-', color= colormap_i[i])
            ax4.plot(cycleIt.getPointTimeSequenceData("LHipPower")[:,2], '-', color= colormap_i[i])

            ax5.plot(cycleIt.getPointTimeSequenceData("LKneeMoment")[:,0], '-', color= colormap_i[i])
            ax6.plot(cycleIt.getPointTimeSequenceData("LKneeMoment")[:,1], '-', color= colormap_i[i])
            ax7.plot(cycleIt.getPointTimeSequenceData("LKneeMoment")[:,2], '-', color= colormap_i[i])
            ax8.plot(cycleIt.getPointTimeSequenceData("LKneePower")[:,2], '-', color= colormap_i[i])

            ax9.plot(cycleIt.getPointTimeSequenceData("LAnkleMoment")[:,0], '-', color= colormap_i[i])
            ax10.plot(cycleIt.getPointTimeSequenceData("LAnkleMoment")[:,1], '-', color= colormap_i[i])
            ax11.plot(cycleIt.getPointTimeSequenceData("LAnkleMoment")[:,2], '-', color= colormap_i[i])
            ax12.plot(cycleIt.getPointTimeSequenceData("LAnklePower")[:,2], '-', color= colormap_i[i])

        if cycleIt.context == "Right":
            ax1.plot(cycleIt.getPointTimeSequenceData("RHipMoment")[:,0], '-', color= colormap_i[i])
            ax2.plot(cycleIt.getPointTimeSequenceData("RHipMoment")[:,1], '-', color= colormap_i[i])
            ax3.plot(cycleIt.getPointTimeSequenceData("RHipMoment")[:,2], '-', color= colormap_i[i])
            ax4.plot(cycleIt.getPointTimeSequenceData("RHipPower")[:,2], '-', color= colormap_i[i])

            ax5.plot(cycleIt.getPointTimeSequenceData("RKneeMoment")[:,0], '-', color= colormap_i[i])
            ax6.plot(cycleIt.getPointTimeSequenceData("RKneeMoment")[:,1], '-', color= colormap_i[i])
            ax7.plot(cycleIt.getPointTimeSequenceData("RKneeMoment")[:,2], '-', color= colormap_i[i])
            ax8.plot(cycleIt.getPointTimeSequenceData("RKneePower")[:,2], '-', color= colormap_i[i])

            ax9.plot(cycleIt.getPointTimeSequenceData("RAnkleMoment")[:,0], '-', color= colormap_i[i])
            ax10.plot(cycleIt.getPointTimeSequenceData("RAnkleMoment")[:,1], '-', color= colormap_i[i])
            ax11.plot(cycleIt.getPointTimeSequenceData("RAnkleMoment")[:,2], '-', color= colormap_i[i])
            ax12.plot(cycleIt.getPointTimeSequenceData("RAnklePower")[:,2], '-', color= colormap_i[i])

        i+=1

    for axIt in axes:
        axIt.set_xlim([0,end])
        axIt.tick_params(axis='x', which='major', labelsize=6)
        axIt.tick_params(axis='y', which='major', labelsize=6)

    ax1.legend(labels,bbox_to_anchor=(0., 1.20, 2., .102),
          ncol=len(cycleInstances), mode="expand", borderaxespad=0., fontsize = 5)

    ax1.set_title("Hip extensor Moment" ,size=8)
    ax2.set_title("Hip adductor Moment" ,size=8)
    ax3.set_title("Hip rotation Moment" ,size=8)
    ax4.set_title("Hip power" ,size=8)

    ax5.set_title("Knee extensor Moment" ,size=8)
    ax6.set_title("Knee adductor Moment" ,size=8)
    ax7.set_title("Knee rotation Moment" ,size=8)
    ax8.set_title("Knee power" ,size=8)

    ax9.set_title("Ankle extensor Moment" ,size=8)
    ax10.set_title("Ankle adductor Moment" ,size=8)
    ax11.set_title("Ankle rotation Moment" ,size=8)
    ax12.set_title("Ankle power" ,size=8)

# ---- local plot ------

def gaitDescriptivePlot(figAxis,analysisStructureItem,
                    pointLabel_L,contextPointLabel_L,
                    pointLabel_R, contextPointLabel_R,
                    axis,
                    title, xlabel="", ylabel="",
                    leftLimits=None,
                    rightLimits=None):

    '''

        **Description :** plot descriptive statistical (average and sd corridor) gait traces from a pyCGM2.Processing.analysis.Analysis instance

        :Parameters:
             - `figAxis` (matplotlib::Axis )
             - `analysisStructureItem` (pyCGM2.Processing.analysis.Analysis.Structure) - a Structure item of an Analysis instance built from AnalysisFilter
             - `pointLabel_L` (str) - selected label of the left point
             - `contextPointLabel_L` (str) - context (Left or Right) of pointLabel_L ( generally Left)
             - `pointLabel_R` (str) - selected label of the left point
             - `contextPointLabel_R` (str) - context (Left or Right) of pointLabel_R ( generally Right)
             - `axis` (int) -  column index of the point axis  (choice : 0,1 or 2)
             - `title` (str) - plot title
             - `xlabel` (str) - label of the x-axis
             - `ylabel` (str) - label of the y-axis
             - `leftLimits` (list) - plot  dashed horizontal axes at  specific y-axis values for a left context
             - `rightLimits` (list) - plot  dashed horizontal axes at  specific y-axis values for a right context

        :Return:
            - matplotlib figure

        **Usage**

        .. code:: python

            plot.gaitDescriptivePlot(ax, analysisFilter.analysis.kinematicStats, 
                                     "LHipAngles","Left", 
                                     "RHipAngles","Right",0,
                                     "Pelvis Tilt", ylabel = " angle (deg)")
   '''





    # check if [ pointlabel , context ] in keys of analysisStructureItem
    left_flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel_L and key[1] == contextPointLabel_L:
            left_flag = True if analysisStructureItem.data[pointLabel_L,contextPointLabel_L]["values"] != [] else False


    right_flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel_R and key[1] == contextPointLabel_R:
            right_flag = True if analysisStructureItem.data[pointLabel_R,contextPointLabel_R]["values"] != [] else False


    # plot
    if left_flag:
        mean_L=analysisStructureItem.data[pointLabel_L,contextPointLabel_L]["mean"][:,axis]
        std_L=analysisStructureItem.data[pointLabel_L,contextPointLabel_L]["std"][:,axis]
        figAxis.plot(np.linspace(0,100,101), mean_L, 'r-')
        figAxis.fill_between(np.linspace(0,100,101), mean_L-std_L, mean_L+std_L, facecolor="red", alpha=0.5,linewidth=0)

        if analysisStructureItem.pst !={}:
            stance = analysisStructureItem.pst['stancePhase', 'Left']["mean"]
            double1 = analysisStructureItem.pst['doubleStance1', 'Left']["mean"]
            double2 = analysisStructureItem.pst['doubleStance2', 'Left']["mean"]
            figAxis.axvline(stance,color='r',ls='dashed')
            figAxis.axvline(double1,color='r',ls='dotted')
            figAxis.axvline(stance-double2,color='r',ls='dotted')

        if leftLimits is not None:
            for value in leftLimits:
                figAxis.axhline(value,color='r',ls='dashed')





    if right_flag:
        mean_R=analysisStructureItem.data[pointLabel_R,contextPointLabel_R]["mean"][:,axis]
        std_R=analysisStructureItem.data[pointLabel_R,contextPointLabel_R]["std"][:,axis]
        figAxis.plot(np.linspace(0,100,101), mean_R, 'b-')
        figAxis.fill_between(np.linspace(0,100,101), mean_R-std_R, mean_R+std_R, facecolor="blue", alpha=0.5,linewidth=0)

        if analysisStructureItem.pst !={}:
            stance = analysisStructureItem.pst['stancePhase', 'Right']["mean"]
            double1 = analysisStructureItem.pst['doubleStance1', 'Right']["mean"]
            double2 = analysisStructureItem.pst['doubleStance2', 'Right']["mean"]
            figAxis.axvline(stance,color='b',ls='dashed')
            figAxis.axvline(double1,color='b',ls='dotted')
            figAxis.axvline(stance-double2,color='b',ls='dotted')

        if rightLimits is not None:
            for value in rightLimits:
                figAxis.axhline(value,color='b',ls='dashed')


    figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    figAxis.set_xlabel(xlabel,size=8)
    figAxis.set_ylabel(ylabel,size=8)


def gaitConsistencyPlot( figAxis, analysisStructureItem,  pointLabel_L,contextPointLabel_L, pointLabel_R, contextPointLabel_R, axis, title, xlabel="", ylabel="",
                    leftLimits=None,
                    rightLimits=None):

    '''

        **Description :** plot all cycle gait traces from a pyCGM2.Processing.analysis.Analysis instance

        :Parameters:
             - `figAxis` (matplotlib::Axis )
             - `analysisStructureItem` (pyCGM2.Processing.analysis.Analysis.Structure) - a Structure item of an Analysis instance built from AnalysisFilter
             - `pointLabel_L` (str) - selected label of the left point
             - `contextPointLabel_L` (str) - context (Left or Right) of pointLabel_L ( generally Left)
             - `pointLabel_R` (str) - selected label of the left point
             - `contextPointLabel_R` (str) - context (Left or Right) of pointLabel_R ( generally Right)
             - `axis` (int) -  column index of the point axis  (choice : 0,1 or 2)
             - `title` (str) - plot title
             - `xlabel` (str) - label of the x-axis
             - `ylabel` (str) - label of the y-axis
             - `leftLimits` (list) - list of double plotting a dashed horizontal axis at a specfic y-axis for a left context
             - `rightLimits` (list) - list of double plotting a dashed horizontal axis at a specfic y-axis for a right context



        :Return:
            - matplotlib figure

        **Usage**

        .. code:: python

            gaitConsistencyPlot(ax1,  analysis, "LPelvisAngles" ,"Left", "RPelvisAngles","Right", 0,
                                "Pelvis Tilt", ylabel = " angle (deg)",
                                leftLimits = [ 25, 35 ],rightLimits = [ 25, 35 ] )


   '''


    # Left plot
    #------------
    # check pointLabel, contextpoint exist
    left_flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel_L and key[1] == contextPointLabel_L:
           n = len(analysisStructureItem.data[pointLabel_L,contextPointLabel_L]["values"])
           left_flag = True if n !=0 else False

    # plot
    if left_flag:
        values_L= np.zeros((101,n))
        i=0
        for val in analysisStructureItem.data[pointLabel_L,contextPointLabel_L]["values"]:
            values_L[:,i] = val[:,axis]
            i+=1

        figAxis.plot(np.linspace(0,100,101), values_L, 'r-')

        if analysisStructureItem.pst !={}:
            for valStance,valDouble1,valDouble2, in zip(analysisStructureItem.pst['stancePhase', 'Left']["values"],analysisStructureItem.pst['doubleStance1', 'Left']["values"],analysisStructureItem.pst['doubleStance2', 'Left']["values"]):
                figAxis.axvline(valStance,color='r',ls='dashed')
                figAxis.axvline(valDouble1,color='r',ls='dotted')
                figAxis.axvline(valStance-valDouble2,color='r',ls='dotted')


        if leftLimits is not None:
            for value in leftLimits:
                figAxis.axhline(value,color='r',ls='dashed')


    # right plot
    #------------
    # check
    right_flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel_R and key[1] == contextPointLabel_R:
            n = len(analysisStructureItem.data[pointLabel_R,contextPointLabel_R]["values"])
            right_flag = True if n !=0 else False

    # plot
    if right_flag:
        values_R= np.zeros((101,n))
        i=0
        for val in analysisStructureItem.data[pointLabel_R,contextPointLabel_R]["values"]:
            values_R[:,i] = val[:,axis]
            i+=1

        figAxis.plot(np.linspace(0,100,101), values_R, 'b-')

        if analysisStructureItem.pst !={}:
            for valStance,valDouble1,valDouble2, in zip(analysisStructureItem.pst['stancePhase', 'Right']["values"],analysisStructureItem.pst['doubleStance1', 'Right']["values"],analysisStructureItem.pst['doubleStance2', 'Right']["values"]):
                figAxis.axvline(valStance,color='b',ls='dashed')
                figAxis.axvline(valDouble1,color='b',ls='dotted')
                figAxis.axvline(valStance-valDouble2,color='b',ls='dotted')

        if rightLimits is not None:
            for value in rightLimits:
                figAxis.axhline(value,color='b',ls='dashed')



    figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100.0])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    figAxis.set_xlabel(xlabel,size=8)
    figAxis.set_ylabel(ylabel,size=8)


# ------ FILTER -----------
class PlottingFilter(object):

    def __init__(self):
        '''
            **Description :** Filter constructor. This filter calls a concrete PlotBuilder and run  its plot-embedded method
        '''
        self.__concretePlotBuilder = None
        self.m_path = None
        self.m_pdfSuffix = None

    def setPath(self,path):
        '''
            **Description :** define path  of the desired output folder

            :Parameters:
             - `path` (str) - path must end with \\


        '''

        self.m_path = path

    def setPdfName(self,name):
        '''
            **Description :** set filename of the pdf

            :Parameters:
             - `name` (str)


        '''

        self.m_pdfName = name



    def setBuilder(self,concretePlotBuilder):
        '''
            **Description :** load a concrete plot builder

            :Parameters:
             - `concretePlotBuilder` (pyCGM2.Report.plot PlotBuilder) - concrete plot builder from pyCGM2.Report.plot module

        '''

        self.__concretePlotBuilder = concretePlotBuilder

    def plot(self):
        '''
            **Description :** Generate plot panels

        '''

        self.__concretePlotBuilder.plotPanel(self.m_path,self.m_pdfName)


# ------ BUILDERS -----------
class AbstractPlotBuilder(object):
    """
    Abstract Builder
    """
    def __init__(self,iObj=None):
        self.m_input =iObj

    def plotPanel(self):
        pass


class StaticAnalysisPlotBuilder(AbstractPlotBuilder):
    def __init__(self,iStaticAnalysis,pointLabelSuffix="",staticModelledFilename=""):
        """
            **Description :** Constructor of static analysis plots

            :Parameters:
                 - `iStaticAnalysis` (pyCGM2.Processing.analysis.StaticAnalysis ) - StaticAnalysis instance built from StaticAnalysisFilter
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels
                 - `staticModelledFilename` (str) - filename of the static c3d including kinematic ouput

            .. warning:: the plotPanel method calls vicon point nomenclature (ex: LHipAngles,LHipMoment,...)

        """

        # TODO :  use a dictionaray for handling point label.( as i did for GaitAnalysisPlotBuilder)

        super(StaticAnalysisPlotBuilder, self).__init__(iObj=iStaticAnalysis)
        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_staticModelledFilename = staticModelledFilename[:-4]

    def plotPanel(self,path,pdfName):
        """
            **Description :** plot the static angle profile

            :Parameters:
                - `path` (str) - path
                - `pdfName` (str) - filename of pdf

            :return:
                - pdf of the static angle profile plot ( ex: "staticAngleProfiles - static cal 01.pdf") )

        """



        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

        pdfNamePlus = "_" + pdfName if pdfName != None else ""

        path = path if  path != None else ""

        pelvis_ante =  self.m_input.data[( self.m_input.data.Label=="LPelvisAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()

        leftHip_x =  self.m_input.data[( self.m_input.data.Label=="LHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()
        rightHip_x =  self.m_input.data[( self.m_input.data.Label=="RHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()

        leftknee_x =  self.m_input.data[( self.m_input.data.Label=="LKneeAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()
        rightknee_x =  self.m_input.data[( self.m_input.data.Label=="RKneeAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()

        leftAnkle_x =  self.m_input.data[( self.m_input.data.Label=="LAnkleAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()
        rightAnkle_x =  self.m_input.data[( self.m_input.data.Label=="RAnkleAngles"+suffixPlus) &  ( self.m_input.data.Axe=="X")  ].Mean.as_matrix()


        pelvis_up =  self.m_input.data[( self.m_input.data.Label=="LPelvisAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Y")  ].Mean.as_matrix()

        leftHip_y =  self.m_input.data[( self.m_input.data.Label=="LHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Y")  ].Mean.as_matrix()
        rightHip_y =  self.m_input.data[( self.m_input.data.Label=="RHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Y")  ].Mean.as_matrix()

        pelvis_int =  self.m_input.data[( self.m_input.data.Label=="LPelvisAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Z")  ].Mean.as_matrix()

        leftHip_z =  self.m_input.data[( self.m_input.data.Label=="LHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Z")  ].Mean.as_matrix()
        rightHip_z =  self.m_input.data[( self.m_input.data.Label=="RHipAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Z")  ].Mean.as_matrix()


        leftFootProgress_z =  self.m_input.data[( self.m_input.data.Label=="LFootProgressAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Z")  ].Mean.as_matrix()
        rightFootProgress_z =  self.m_input.data[( self.m_input.data.Label=="RFootProgressAngles"+suffixPlus) &  ( self.m_input.data.Axe=="Z")  ].Mean.as_matrix()

#        fig, ax = plt.subplots(figsize=(8.27,3.93), dpi=100,facecolor="white")

        fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        ax = plt.subplot(5,1,1)

        width = 1
        plt.bar(0,pelvis_ante,width, color = "g", label = "overall")

        plt.bar(1.5,leftHip_x,width, color = "r", label = "left")
        plt.bar(2.5,rightHip_x,width, color = "b", label = "right")

        plt.bar(4,leftknee_x,width, color = "r")
        plt.bar(5,rightknee_x,width, color = "b")

        plt.bar(6.5,leftAnkle_x,width, color = "r")
        plt.bar(7.5,rightAnkle_x,width, color = "b")

        plt.bar(9,pelvis_up,width, color = "g")

        plt.bar(10.5,leftHip_y,width, color = "r")
        plt.bar(11.5,rightHip_y,width, color = "b")

        plt.bar(13,pelvis_int,width, color = "g")

        plt.bar(14.5,leftHip_z,width, color = "r")
        plt.bar(15.5,rightHip_z,width, color = "b")


        plt.bar(17,leftFootProgress_z,width, color = "r")
        plt.bar(18,rightFootProgress_z,width, color = "b")

        ax.set_xticks([0.5,
                       2.5,
                       5,
                       7.5,
                       9.5,
                       11.5,
                       13.5,
                       15.5,
                       18
                       ])
        ax.set_xticklabels(["Pelvis (ante/pos)",
                            "Hip (flexion/ext)",
                            "Knee (flexion/ext)",
                            "Ankle (flexion/ext)",
                            "Pelvis (obli)",
                            "Hip (abd/Add)",
                            "Pelvis (int/ext)",
                            "Hip (int/ext)",
                            "Foot Progress (int/ext)"],rotation=30,size = 6)

        red_patch = mpatches.Patch(color='red', label='Left')
        blue_patch = mpatches.Patch(color='blue', label='Right')
        green_patch = mpatches.Patch(color='green', label='Overall')

        ax.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(0., 1.2, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,fontsize = 8)



        ax.set_title(self.m_staticModelledFilename + " - Static Angle Profile" ,size=12)

        ax.tick_params(axis='y', which='major', labelsize=6)

        pp = PdfPages(str(path+ "staticAngleProfiles"+pdfNamePlus+".pdf"))
        pp.savefig(fig)
        pp.close()



class GaitAnalysisPlotBuilder(AbstractPlotBuilder):
    def __init__(self,iAnalysis,kineticFlag=True,pointLabelSuffix=""):

        """
            **Description :** Constructor of gait plot panel.

            .. warning:: 
            
                By default, the plotPanel method uses vicon point nomenclature through a dictionnary ``translators``. 
                Use *setTranslator* method if you want to use another point label 


            :Parameters:
                 - `iAnalysis` (pyCGM2.Processing.analysis.Analysis ) - Analysis instance built from AnalysisFilter
                 - `kineticFlag` (bool) - enable kinetic plots
                 - `pointLabelSuffix` (str) - suffix ending conventional kinetic CGM labels


            .. note::

                The kinematic panel is made of 12 subplots
    
                ================  ==============================    ===========
                matplotlib Axis   translators                       Axis label
                ================  ==============================    ===========
                ax1               "Left.PelvisProgress.Angles"      Tilt
                                  "Right.PelvisProgress.Angles"
                ax2               "Left.PelvisProgress.Angles"      Obli
                                  "Right.PelvisProgress.Angles"
                ax3               "Left.PelvisProgress.Angles"      Rota
                                  "Right.PelvisProgress.Angles"
                ax4               "Left.Hip.Angles"                 Flex
                                  "Right.Hip.Angles"
                ax5               "Left.Hip.Angles"                 Addu
                                  "Right.Hip.Angles"
                ax6               "Left.Hip.Angles"                 Rota
                                  "Right.Hip.Angles"
                ax7               "Left.Knee.Angles"                Flex
                                  "Right.Knee.Angles"
                ax8               "Left.Knee.Angles"                Addu
                                  "Right.Knee.Angles"
                ax9               "Left.Knee.Angles"                Rota
                                  "Right.Knee.Angles"
                ax10              "Left.Ankle.Angles"               Flex
                                  "Right.Ankle.Angles"
                ax11              "Left.Ankle.Angles"               Addu
                                  "Right.Ankle.Angles"
                ax12              "Left.FootProgress.Angles"        Rota
                                  "Right.FootProgress.Angles"
                ================  ==============================    ===========
    
                The kinetic panel is made of 12 subplots
    
                ================  ==============================    ===========
                matplotlib Axis   translators                       Axis label
                ================  ==============================    ===========
                ax1               "Left.Hip.Moment"                 Ext
                                  "Right.Hip.Moment"
                ax2               "Left.Hip.Moment"                 Abd
                                  "Right.Hip.Moment"
                ax3               "Left.Hip.Moment"                 Rot
                                  "Right.Hip.Moment"
                ax4               "Left.Hip.Power"
                                  "Right.Hip.Power"
                ax5               "Left.Knee.Moment"                Ext
                                  "Right.Knee.Moment"
                ax6               "Left.Knee.Moment"                Abd
                                  "Right.Knee.Moment"
                ax7               "Left.Knee.Moment"                Rot
                                  "Right.Knee.Moment"
                ax8               "Left.Knee.Power"                  
                                  "Right.Knee.Power"
                ax9               "Left.Ankle.Moment"               Pla
                                  "Right.Ankle.Moment"
                ax10              "Left.Ankle.Moment"               Rot
                                  "Right.Ankle.Moment"
                ax11              "Left.Ankle.Moment"               Eve
                                  "Right.Ankle.Moment"
                ax12              "Left.Ankle.Power"
                                  "Right.Ankle.Power"
                ================  ==============================    ===========

        """


        super(GaitAnalysisPlotBuilder, self).__init__(iObj=iAnalysis)

        if isinstance(self.m_input,CGM2analysis.Analysis):
            pass
        else:
            logging.error( "[pyCGM2] error input object type. must be a pyCGM2.Core.Processing.analysis.Analysis")

        self.m_pointLabelSuffix = pointLabelSuffix
        self.m_kineticFlag = kineticFlag

        self.m_nd_procedure = None
        self.m_flagConsistencyOnly = False

        self.__translators=dict()
        self.__translators["Left.PelvisProgress.Angles"] = "LPelvisAngles"
        self.__translators["Left.Hip.Angles"] = "LHipAngles"
        self.__translators["Left.Knee.Angles"] = "LKneeAngles"
        self.__translators["Left.Ankle.Angles"] = "LAnkleAngles"
        self.__translators["Left.FootProgress.Angles"] = "LFootProgressAngles"

        self.__translators["Right.PelvisProgress.Angles"] = "RPelvisAngles"
        self.__translators["Right.Hip.Angles"] = "RHipAngles"
        self.__translators["Right.Knee.Angles"] = "RKneeAngles"
        self.__translators["Right.Ankle.Angles"] = "RAnkleAngles"
        self.__translators["Right.FootProgress.Angles"] = "RFootProgressAngles"

        self.__translators["Left.Hip.Moment"] = "LHipMoment"
        self.__translators["Left.Knee.Moment"] = "LKneeMoment"
        self.__translators["Left.Ankle.Moment"] = "LAnkleMoment"

        self.__translators["Right.Hip.Moment"] = "RHipMoment"
        self.__translators["Right.Knee.Moment"] = "RKneeMoment"
        self.__translators["Right.Ankle.Moment"] = "RAnkleMoment"


        self.__translators["Left.Hip.Power"] = "LHipPower"
        self.__translators["Left.Knee.Power"] = "LKneePower"
        self.__translators["Left.Ankle.Power"] = "LAnklePower"

        self.__translators["Right.Hip.Power"] = "RHipPower"
        self.__translators["Right.Knee.Power"] = "RKneePower"
        self.__translators["Right.Ankle.Power"] = "RAnklePower"

        self.__limits=dict()
        for side in ["Left","Right"]:
            self.__limits[str(side)+".PelvisProgress.Angles","Tilt"] = None
            self.__limits[str(side)+".PelvisProgress.Angles","Obli"] = None
            self.__limits[str(side)+".PelvisProgress.Angles","Rota"] = None

            self.__limits[str(side)+".Hip.Angles","Flex"] = None
            self.__limits[str(side)+".Hip.Angles","Addu"] = None
            self.__limits[str(side)+".Hip.Angles","Rota"] = None

            self.__limits[str(side)+".Knee.Angles","Flex"] = None
            self.__limits[str(side)+".Knee.Angles","Addu"] = None
            self.__limits[str(side)+".Knee.Angles","Rota"] = None

            self.__limits[str(side)+".Ankle.Angles","Flex"] = None
            self.__limits[str(side)+".Ankle.Angles","Addu"] = None

            self.__limits[str(side)+".FootProgress.Angles","Rota"] = None


            self.__limits[str(side)+".Hip.Moment","Ext"] = None
            self.__limits[str(side)+".Hip.Moment","Abd"] = None
            self.__limits[str(side)+".Hip.Moment","Rot"] = None
            self.__limits[str(side)+".Hip.Power"] = None

            self.__limits[str(side)+".Knee.Moment","Ext"] = None
            self.__limits[str(side)+".Knee.Moment","Abd"] = None
            self.__limits[str(side)+".Knee.Moment","Rot"] = None
            self.__limits[str(side)+".Knee.Power"] = None

            self.__limits[str(side)+".Ankle.Moment","Pla"] = None
            self.__limits[str(side)+".Ankle.Moment","Rot"] = None
            self.__limits[str(side)+".Ankle.Moment","Eve"] = None
            self.__limits[str(side)+".Ankle.Power"] = None


    def printTranslators(self):
        """
            **Description :** display all translator labels
        """
        for key in self.__translators.keys():
            print key


    def setTranslators(self,keys,newNames):

        """
            **Description :** set all translators

            :Parameters:

                 - `keys` (list) - list of translator dictionnary items
                 - `newNames` (list) - list of new labels

        """

        if not isinstance(newNames,list) or not isinstance(keys,list):
            raise Exception ( "[pyCGM2] input parameter must be a list ")
        else:
            if len(keys) != len(newNames):
                raise Exception( "[pyCGM2] input argument unbalanced")
            else:
                for key,name in zip(keys,newNames):
                    if key not in self.__translators.keys():
                        raise Exception( "key %s doesn t exit")
                    else:
                        self.__translators[key] = name

    def setTranslator(self,key,newName):
        """
            **Description :** set a translator

            :Parameters:

                 - `key` (str) - name of a translator dictionnary item
                 - `newName` (str) - label to attribute

        """

        if key not in self.__translators.keys():
            raise Exception( "[pyCGM2] key %s doesn t exit")
        else:
            self.__translators[key] = newName


    def setLimits(self,keyLabel,keyAxis,values):
        """
            **Description :** set dashed horizintal line for combinaison  traslator label - axis Label

            :Parameters:
                - `keyLabel` (str) - name of a translator dictionnary item
                - `keyAxis` (str) - axis label  of the point  ( depend on the selected joint, see __init__ table )
                - `values` (list of double) . list of y- axis values


            **Usage**

            .. code:: python

                plotBuilder.setLimits("Left.Knee.Angles","Flex" , [40,30,25])

        """
        if not isinstance(values,list):
            raise Exception( "[pyCGM2] input values must be a list ")
        else:
            for key in self.__limits.keys():
                flag = False
                if key[0] == keyLabel and key[1]==keyAxis:
                    self.__limits[keyLabel,keyAxis] = values
                    flag = True
                    break
            if not flag:
                raise Exception( "[pyCGM2] check your input ( keyAngle or keyAxis not found) ")


  
    def setNormativeDataProcedure(self,ndp):
        """
            **Description :** set a normative gait dataset

            :Parameters:
                 - `ndp` (a class of the pyCGM2.Report.normativeDatabaseProcedure module) - normative gait dataset from pyCGM2.Report.normativeDatabaseProcedure module

        """

        self.m_nd_procedure = ndp
        self.m_nd_procedure.constructNormativeData()

    def setConsistencyOnly(self,iBool):
        """
            **Description :** only consistency panel will be plot

            :Parameters:
                 - `iBool` (bool)

        """

        self.m_flagConsistencyOnly = iBool


    def plotPanel(self,path,pdfName):

        """
            **Description :** plot gait panels

            :Parameters:
                - `path` (str) - path
                - `pdfName` (str) - filename of pdf

            :return:
                - pdf of descriptiveKinematics plots
                - pdf of descriptiveKinetics plots
                - pdf of consistencyKinematics plots
                - pdf of consistencyKinetics plots
        """

        pdfNamePlus = "_" + pdfName if pdfName != None else ""

        path = path if  path != None else ""


        if self.m_nd_procedure is not None:
            normativeData = self.m_nd_procedure.data
        else:
            normativeData = None

        if not self.m_flagConsistencyOnly:
            # ---- descriptive Kinematics (mean + std)
            figDescritiveKinematics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
            title=u""" Descriptive Kinematics \n """
            figDescritiveKinematics.suptitle(title)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

            ax1 = plt.subplot(5,3,1)# Pelvis X
            ax2 = plt.subplot(5,3,2)# Pelvis Y
            ax3 = plt.subplot(5,3,3)# Pelvis Z
            ax4 = plt.subplot(5,3,4)# Hip X
            ax5 = plt.subplot(5,3,5)# Hip Y
            ax6 = plt.subplot(5,3,6)# Hip Z
            ax7 = plt.subplot(5,3,7)# Knee X
            ax8 = plt.subplot(5,3,8)# Knee Y
            ax9 = plt.subplot(5,3,9)# Knee Z
            ax10 = plt.subplot(5,3,10)# Ankle X
            ax11 = plt.subplot(5,3,11)# Ankle Y
            ax12 = plt.subplot(5,3,12)# Footprogress Z


            axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

            # make
            suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""
            gaitDescriptivePlot(ax1,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",0,"Pelvis Tilt", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Tilt"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Tilt"])
                            
            gaitDescriptivePlot(ax2,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",1,"Pelvis Obliquity", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Obli"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Obli"])
                          
            gaitDescriptivePlot(ax3,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",2,"Pelvis Rotation", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Rota"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Rota"])
                          
            gaitDescriptivePlot(ax4,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",0,"Hip flexion", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Flex"],rightLimits = self.__limits["Right.Hip.Angles","Flex"])
                          
            gaitDescriptivePlot(ax5,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",1,"Hip adduction",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Addu"],rightLimits = self.__limits["Right.Hip.Angles","Addu"])
                          
            gaitDescriptivePlot(ax6,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",2,"Hip rotation",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Hip.Angles","Rota"],rightLimits = self.__limits["Right.Hip.Angles","Rota"])
                          
            gaitDescriptivePlot(ax7,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",0,"Knee flexion",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Flex"],rightLimits = self.__limits["Right.Knee.Angles","Flex"])
                          
            gaitDescriptivePlot(ax8,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",1,"Knee adduction",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Addu"],rightLimits = self.__limits["Right.Knee.Angles","Addu"])
                          
            gaitDescriptivePlot(ax9,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",2,"Knee rotation",ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.Knee.Angles","Rota"],rightLimits = self.__limits["Right.Knee.Angles","Rota"])

            gaitDescriptivePlot(ax10, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",0,"Ankle dorsiflexion",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.Ankle.Angles","Flex"],rightLimits = self.__limits["Right.Ankle.Angles","Flex"])

            gaitDescriptivePlot(ax11, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",1,"Ankle adduction",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.Ankle.Angles","Addu"],rightLimits = self.__limits["Right.Ankle.Angles","Addu"])

            gaitDescriptivePlot(ax12, self.m_input.kinematicStats,  self.__translators["Left.FootProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.FootProgress.Angles"]+suffixPlus,"Right",1,"Foot Progression",ylabel = "angle (deg)", xlabel ="Gait cycle %",
                            leftLimits = self.__limits["Left.FootProgress.Angles","Rota"],rightLimits = self.__limits["Right.FootProgress.Angles","Rota"])



            if normativeData is not None:
                ax1.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,0]-normativeData["Pelvis.Angles"]["sd"][:,0], normativeData["Pelvis.Angles"]["mean"][:,0]+normativeData["Pelvis.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax2.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,1]-normativeData["Pelvis.Angles"]["sd"][:,1], normativeData["Pelvis.Angles"]["mean"][:,1]+normativeData["Pelvis.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                ax3.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,2]-normativeData["Pelvis.Angles"]["sd"][:,2], normativeData["Pelvis.Angles"]["mean"][:,2]+normativeData["Pelvis.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
                ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,0]-normativeData["Hip.Angles"]["sd"][:,0], normativeData["Hip.Angles"]["mean"][:,0]+normativeData["Hip.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax5.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,1]-normativeData["Hip.Angles"]["sd"][:,1], normativeData["Hip.Angles"]["mean"][:,1]+normativeData["Hip.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
                ax6.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,2]-normativeData["Hip.Angles"]["sd"][:,2], normativeData["Hip.Angles"]["mean"][:,2]+normativeData["Hip.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
                ax7.fill_between(np.linspace(0,100,51), normativeData["Knee.Angles"]["mean"][:,0]-normativeData["Knee.Angles"]["sd"][:,0], normativeData["Knee.Angles"]["mean"][:,0]+normativeData["Knee.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,0]-normativeData["Ankle.Angles"]["sd"][:,0], normativeData["Ankle.Angles"]["mean"][:,0]+normativeData["Ankle.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,2]-normativeData["Ankle.Angles"]["sd"][:,2], normativeData["Ankle.Angles"]["mean"][:,2]+normativeData["Ankle.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)


            pp = PdfPages(str(path+ "descriptiveKinematics"+pdfNamePlus+".pdf"))
            pp.savefig(figDescritiveKinematics)
            pp.close()


        # ---- consitency Kinematics plot
        figConsistencyKinematics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
        title=u""" Consistency Kinematics \n """
        figConsistencyKinematics.suptitle(title)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

        ax1 = plt.subplot(5,3,1)# Pelvis X
        ax2 = plt.subplot(5,3,2)# Pelvis Y
        ax3 = plt.subplot(5,3,3)# Pelvis Z
        ax4 = plt.subplot(5,3,4)# Hip X
        ax5 = plt.subplot(5,3,5)# Hip Y
        ax6 = plt.subplot(5,3,6)# Hip Z
        ax7 = plt.subplot(5,3,7)# Knee X
        ax8 = plt.subplot(5,3,8)# Knee Y
        ax9 = plt.subplot(5,3,9)# Knee Z
        ax10 = plt.subplot(5,3,10)# Ankle X
        ax11 = plt.subplot(5,3,11)# Ankle Y
        ax12 = plt.subplot(5,3,12)# FootProgress


        axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

        # make
        suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""
        gaitConsistencyPlot(ax1,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",0,"Pelvis Tilt", ylabel = " angle (deg)",
                            leftLimits = self.__limits["Left.PelvisProgress.Angles","Tilt"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Tilt"])

        gaitConsistencyPlot(ax2,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",1,"Pelvis Obliquity", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.PelvisProgress.Angles","Obli"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Obli"])

        gaitConsistencyPlot(ax3,  self.m_input.kinematicStats, self.__translators["Left.PelvisProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.PelvisProgress.Angles"]+suffixPlus,"Right",2,"Pelvis Rotation", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.PelvisProgress.Angles","Rota"],rightLimits = self.__limits["Right.PelvisProgress.Angles","Rota"])

        gaitConsistencyPlot(ax4,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",0,"Hip flexion", ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Flex"],rightLimits = self.__limits["Right.Hip.Angles","Flex"])

        gaitConsistencyPlot(ax5,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",1,"Hip adduction",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Addu"],rightLimits = self.__limits["Right.Hip.Angles","Addu"])

        gaitConsistencyPlot(ax6,  self.m_input.kinematicStats, self.__translators["Left.Hip.Angles"]+suffixPlus,"Left", self.__translators["Right.Hip.Angles"]+suffixPlus,"Right",2,"Hip rotation",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Hip.Angles","Rota"],rightLimits = self.__limits["Right.Hip.Angles","Rota"])

        gaitConsistencyPlot(ax7,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",0,"Knee flexion",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Flex"],rightLimits = self.__limits["Right.Knee.Angles","Flex"])

        gaitConsistencyPlot(ax8,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",1,"Knee adduction",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Addu"],rightLimits = self.__limits["Right.Knee.Angles","Addu"])

        gaitConsistencyPlot(ax9,  self.m_input.kinematicStats, self.__translators["Left.Knee.Angles"]+suffixPlus,"Left", self.__translators["Right.Knee.Angles"]+suffixPlus,"Right",2,"Knee rotation",ylabel = " angle (deg)",
                        leftLimits = self.__limits["Left.Knee.Angles","Rota"],rightLimits = self.__limits["Right.Knee.Angles","Rota"])

        gaitConsistencyPlot(ax10, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",0,"Ankle dorsiflexion",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.Ankle.Angles","Flex"],rightLimits = self.__limits["Right.Ankle.Angles","Flex"])

        gaitConsistencyPlot(ax11, self.m_input.kinematicStats,  self.__translators["Left.Ankle.Angles"]+suffixPlus,"Left", self.__translators["Right.Ankle.Angles"]+suffixPlus,"Right",1,"Ankle adduction",ylabel = " angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.Ankle.Angles","Addu"],rightLimits = self.__limits["Right.Ankle.Angles","Addu"])

        gaitConsistencyPlot(ax12, self.m_input.kinematicStats,  self.__translators["Left.FootProgress.Angles"]+suffixPlus,"Left", self.__translators["Right.FootProgress.Angles"]+suffixPlus,"Right",1,"Foot Progression",ylabel = "angle (deg)", xlabel ="Gait cycle %",
                        leftLimits = self.__limits["Left.FootProgress.Angles","Rota"],rightLimits = self.__limits["Right.FootProgress.Angles","Rota"])


        if normativeData is not None:
            ax1.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,0]-normativeData["Pelvis.Angles"]["sd"][:,0], normativeData["Pelvis.Angles"]["mean"][:,0]+normativeData["Pelvis.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax2.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,1]-normativeData["Pelvis.Angles"]["sd"][:,1], normativeData["Pelvis.Angles"]["mean"][:,1]+normativeData["Pelvis.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
            ax3.fill_between(np.linspace(0,100,51), normativeData["Pelvis.Angles"]["mean"][:,2]-normativeData["Pelvis.Angles"]["sd"][:,2], normativeData["Pelvis.Angles"]["mean"][:,2]+normativeData["Pelvis.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
            ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,0]-normativeData["Hip.Angles"]["sd"][:,0], normativeData["Hip.Angles"]["mean"][:,0]+normativeData["Hip.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax5.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,1]-normativeData["Hip.Angles"]["sd"][:,1], normativeData["Hip.Angles"]["mean"][:,1]+normativeData["Hip.Angles"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)
            ax6.fill_between(np.linspace(0,100,51), normativeData["Hip.Angles"]["mean"][:,2]-normativeData["Hip.Angles"]["sd"][:,2], normativeData["Hip.Angles"]["mean"][:,2]+normativeData["Hip.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)
            ax7.fill_between(np.linspace(0,100,51), normativeData["Knee.Angles"]["mean"][:,0]-normativeData["Knee.Angles"]["sd"][:,0], normativeData["Knee.Angles"]["mean"][:,0]+normativeData["Knee.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

            ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,0]-normativeData["Ankle.Angles"]["sd"][:,0], normativeData["Ankle.Angles"]["mean"][:,0]+normativeData["Ankle.Angles"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
            ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Angles"]["mean"][:,2]-normativeData["Ankle.Angles"]["sd"][:,2], normativeData["Ankle.Angles"]["mean"][:,2]+normativeData["Ankle.Angles"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)


        pp = PdfPages(str(path+ "consistencyKinematics"+pdfNamePlus+".pdf"))
        pp.savefig(figConsistencyKinematics)
        pp.close()

        if self.m_kineticFlag:
            if not self.m_flagConsistencyOnly:
                # ---- descriptive Kinetics (mean + std)
                figDescriptiveKinetics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
                title=u""" Descriptive Kinetics \n """
                figDescriptiveKinetics.suptitle(title)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

                ax1 = plt.subplot(3,4,1)# Hip X extensor
                ax2 = plt.subplot(3,4,2)# Hip Y abductor
                ax3 = plt.subplot(3,4,3)# Hip Z rotation
                ax4 = plt.subplot(3,4,4)# Knee Z power

                ax5 = plt.subplot(3,4,5)# knee X extensor
                ax6 = plt.subplot(3,4,6)# knee Y abductor
                ax7 = plt.subplot(3,4,7)# knee Z rotation
                ax8 = plt.subplot(3,4,8)# knee Z power

                ax9 = plt.subplot(3,4,9)# ankle X plantar flexion
                ax10 = plt.subplot(3,4,10)# ankle Y rotation
                ax11 = plt.subplot(3,4,11)# ankle Z everter
                ax12 = plt.subplot(3,4,12)# ankle Z power


                axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

                #  make
                suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

                gaitDescriptivePlot(ax1,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",0,"Hip extensor Moment", ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Ext"],rightLimits = self.__limits["Right.Hip.Moment","Ext"])
                gaitDescriptivePlot(ax2,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",1,"Hip abductor Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Abd"],rightLimits = self.__limits["Right.Hip.Moment","Abd"])
                gaitDescriptivePlot(ax3,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",2,"Hip rotation Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Rot"],rightLimits = self.__limits["Right.Hip.Moment","Rot"])
                gaitDescriptivePlot(ax4,self.m_input.kineticStats, self.__translators["Left.Hip.Power"]+suffixPlus, "Left", self.__translators["Right.Hip.Power"]+suffixPlus,"Right",2,"Hip power",ylabel = " power (W.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Power"],rightLimits = self.__limits["Right.Hip.Power"])


                gaitDescriptivePlot(ax5,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",0,"Knee extensor Moment", ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Knee.Moment","Ext"],rightLimits = self.__limits["Right.Knee.Moment","Ext"])
                gaitDescriptivePlot(ax6,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",1,"Knee abductor Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Knee.Moment","Abd"],rightLimits = self.__limits["Right.Knee.Moment","Abd"])
                gaitDescriptivePlot(ax7,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",2,"Knee rotation Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Knee.Moment","Rot"],rightLimits = self.__limits["Right.Knee.Moment","Rot"])
                gaitDescriptivePlot(ax8,self.m_input.kineticStats, self.__translators["Left.Knee.Power"]+suffixPlus, "Left", self.__translators["Right.Knee.Power"]+suffixPlus,"Right",2,"Knee power",ylabel = " power (W.kg-1)",
                                    leftLimits = self.__limits["Left.Knee.Power"],rightLimits = self.__limits["Right.Knee.Power"])

                gaitDescriptivePlot(ax9,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",0,"Ankle plantarflexor Moment", ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                    leftLimits = self.__limits["Left.Ankle.Moment","Pla"],rightLimits = self.__limits["Right.Ankle.Moment","Pla"])
                gaitDescriptivePlot(ax10,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",1,"Ankle rotation Moment",ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                    leftLimits = self.__limits["Left.Ankle.Moment","Rot"],rightLimits = self.__limits["Right.Ankle.Moment","Rot"])
                gaitDescriptivePlot(ax11,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",2,"Ankle everter Moment",ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                    leftLimits = self.__limits["Left.Ankle.Moment","Eve"],rightLimits = self.__limits["Right.Ankle.Moment","Eve"])
                gaitDescriptivePlot(ax12,self.m_input.kineticStats, self.__translators["Left.Ankle.Power"]+suffixPlus, "Left", self.__translators["Right.Ankle.Power"]+suffixPlus,"Right",2,"Ankle power",ylabel = " power (W.kg-1)",xlabel ="Gait cycle %",
                                    leftLimits = self.__limits["Left.Ankle.Power"],rightLimits = self.__limits["Right.Ankle.Power"])


                if self.m_nd_procedure is not None:

                    ax1.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,0]-normativeData["Hip.Moment"]["sd"][:,0], normativeData["Hip.Moment"]["mean"][:,0]+normativeData["Hip.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax2.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,1]-normativeData["Hip.Moment"]["sd"][:,1], normativeData["Hip.Moment"]["mean"][:,1]+normativeData["Hip.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                    ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Power"]["mean"][:,2]-normativeData["Hip.Power"]["sd"][:,2], normativeData["Hip.Power"]["mean"][:,2]+normativeData["Hip.Power"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                    ax5.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,0]-normativeData["Knee.Moment"]["sd"][:,0], normativeData["Knee.Moment"]["mean"][:,0]+normativeData["Knee.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax6.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,1]-normativeData["Knee.Moment"]["sd"][:,1], normativeData["Knee.Moment"]["mean"][:,1]+normativeData["Knee.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                    ax8.fill_between(np.linspace(0,100,51), normativeData["Knee.Power"]["mean"][:,2]-normativeData["Knee.Power"]["sd"][:,2], normativeData["Knee.Power"]["mean"][:,2]+normativeData["Knee.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

                    ax9.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,0]-normativeData["Ankle.Moment"]["sd"][:,0], normativeData["Ankle.Moment"]["mean"][:,0]+normativeData["Ankle.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                    ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,2]-normativeData["Ankle.Moment"]["sd"][:,1], normativeData["Ankle.Moment"]["mean"][:,1]+normativeData["Ankle.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                    ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Power"]["mean"][:,2]-normativeData["Ankle.Power"]["sd"][:,2], normativeData["Ankle.Power"]["mean"][:,2]+normativeData["Ankle.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)


                pp = PdfPages(str(path+ "descriptiveKinetics"+pdfNamePlus+".pdf"))
                pp.savefig(figDescriptiveKinetics)
                pp.close()


            # ---- consistency Kinetics (mean + std)
            figConsistencyKinetics = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
            title=u""" Consistency Kinetics \n """
            figConsistencyKinetics.suptitle(title)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

            ax1 = plt.subplot(3,4,1)# Hip X extensor
            ax2 = plt.subplot(3,4,2)# Hip Y abductor
            ax3 = plt.subplot(3,4,3)# Hip Z rotation
            ax4 = plt.subplot(3,4,4)# Knee Z power

            ax5 = plt.subplot(3,4,5)# knee X extensor
            ax6 = plt.subplot(3,4,6)# knee Y abductor
            ax7 = plt.subplot(3,4,7)# knee Z rotation
            ax8 = plt.subplot(3,4,8)# knee Z power

            ax9 = plt.subplot(3,4,9)# amkle X plantar flexion
            ax10 = plt.subplot(3,4,10)# ankle Y rotation
            ax11 = plt.subplot(3,4,11)# ankle Z everter
            ax12 = plt.subplot(3,4,12)# ankle Z power


            axes=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

            #  make
            suffixPlus = "_" + self.m_pointLabelSuffix if self.m_pointLabelSuffix!="" else ""

            gaitConsistencyPlot(ax1,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",0,"Hip extensor Moment", ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Ext"],rightLimits = self.__limits["Right.Hip.Moment","Ext"])
            gaitConsistencyPlot(ax2,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",1,"Hip abductor Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Abd"],rightLimits = self.__limits["Right.Hip.Moment","Abd"])
            gaitConsistencyPlot(ax3,self.m_input.kineticStats, self.__translators["Left.Hip.Moment"]+suffixPlus,"Left", self.__translators["Right.Hip.Moment"]+suffixPlus,"Right",2,"Hip rotation Moment",ylabel = " moment (N.mm.kg-1)",
                                    leftLimits = self.__limits["Left.Hip.Moment","Rot"],rightLimits = self.__limits["Right.Hip.Moment","Rot"])
            gaitConsistencyPlot(ax4,self.m_input.kineticStats, self.__translators["Left.Hip.Power"]+suffixPlus, "Left", self.__translators["Right.Hip.Power"]+suffixPlus,"Right",2,"Hip power",ylabel = " power (W.kg-1)",
                                leftLimits = self.__limits["Left.Hip.Power"],rightLimits = self.__limits["Right.Hip.Power"])


            gaitConsistencyPlot(ax5,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",0,"Knee extensor Moment", ylabel = " moment (N.mm.kg-1)",
                                leftLimits = self.__limits["Left.Knee.Moment","Ext"],rightLimits = self.__limits["Right.Knee.Moment","Ext"])
            gaitConsistencyPlot(ax6,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",1,"Knee abductor Moment",ylabel = " moment (N.mm.kg-1)",
                                leftLimits = self.__limits["Left.Knee.Moment","Abd"],rightLimits = self.__limits["Right.Knee.Moment","Abd"])
            gaitConsistencyPlot(ax7,self.m_input.kineticStats, self.__translators["Left.Knee.Moment"]+suffixPlus,"Left", self.__translators["Right.Knee.Moment"]+suffixPlus,"Right",2,"Knee rotation Moment",ylabel = " moment (N.mm.kg-1)",
                                leftLimits = self.__limits["Left.Knee.Moment","Rot"],rightLimits = self.__limits["Right.Knee.Moment","Rot"])
            gaitConsistencyPlot(ax8,self.m_input.kineticStats, self.__translators["Left.Knee.Power"]+suffixPlus, "Left", self.__translators["Right.Knee.Power"]+suffixPlus,"Right",2,"Knee power",ylabel = " power (W.kg-1)",
                                leftLimits = self.__limits["Left.Knee.Power"],rightLimits = self.__limits["Right.Knee.Power"])

            gaitConsistencyPlot(ax9,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",0,"Ankle plantarflexor Moment", ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                leftLimits = self.__limits["Left.Ankle.Moment","Pla"],rightLimits = self.__limits["Right.Ankle.Moment","Pla"])
            gaitConsistencyPlot(ax10,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",1,"Ankle rotation Moment",ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                leftLimits = self.__limits["Left.Ankle.Moment","Rot"],rightLimits = self.__limits["Right.Ankle.Moment","Rot"])
            gaitConsistencyPlot(ax11,self.m_input.kineticStats, self.__translators["Left.Ankle.Moment"]+suffixPlus,"Left", self.__translators["Right.Ankle.Moment"]+suffixPlus,"Right",2,"Ankle everter Moment",ylabel = " moment (N.mm.kg-1)",xlabel ="Gait cycle %",
                                leftLimits = self.__limits["Left.Ankle.Moment","Eve"],rightLimits = self.__limits["Right.Ankle.Moment","Eve"])
            gaitConsistencyPlot(ax12,self.m_input.kineticStats, self.__translators["Left.Ankle.Power"]+suffixPlus, "Left", self.__translators["Right.Ankle.Power"]+suffixPlus,"Right",2,"Ankle power",ylabel = " power (W.kg-1)",xlabel ="Gait cycle %",
                                leftLimits = self.__limits["Left.Ankle.Power"],rightLimits = self.__limits["Right.Ankle.Power"])            
            
            
            if normativeData is not None:
                ax1.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,0]-normativeData["Hip.Moment"]["sd"][:,0], normativeData["Hip.Moment"]["mean"][:,0]+normativeData["Hip.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax2.fill_between(np.linspace(0,100,51), normativeData["Hip.Moment"]["mean"][:,1]-normativeData["Hip.Moment"]["sd"][:,1], normativeData["Hip.Moment"]["mean"][:,1]+normativeData["Hip.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                ax4.fill_between(np.linspace(0,100,51), normativeData["Hip.Power"]["mean"][:,2]-normativeData["Hip.Power"]["sd"][:,2], normativeData["Hip.Power"]["mean"][:,2]+normativeData["Hip.Power"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)

                ax5.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,0]-normativeData["Knee.Moment"]["sd"][:,0], normativeData["Knee.Moment"]["mean"][:,0]+normativeData["Knee.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax6.fill_between(np.linspace(0,100,51), normativeData["Knee.Moment"]["mean"][:,1]-normativeData["Knee.Moment"]["sd"][:,1], normativeData["Knee.Moment"]["mean"][:,1]+normativeData["Knee.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                ax8.fill_between(np.linspace(0,100,51), normativeData["Knee.Power"]["mean"][:,2]-normativeData["Knee.Power"]["sd"][:,2], normativeData["Knee.Power"]["mean"][:,2]+normativeData["Knee.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

                ax9.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,0]-normativeData["Ankle.Moment"]["sd"][:,0], normativeData["Ankle.Moment"]["mean"][:,0]+normativeData["Ankle.Moment"]["sd"][:,0], facecolor="green", alpha=0.5,linewidth=0)
                ax10.fill_between(np.linspace(0,100,51), normativeData["Ankle.Moment"]["mean"][:,2]-normativeData["Ankle.Moment"]["sd"][:,1], normativeData["Ankle.Moment"]["mean"][:,1]+normativeData["Ankle.Moment"]["sd"][:,1], facecolor="green", alpha=0.5,linewidth=0)

                ax12.fill_between(np.linspace(0,100,51), normativeData["Ankle.Power"]["mean"][:,2]-normativeData["Ankle.Power"]["sd"][:,2], normativeData["Ankle.Power"]["mean"][:,2]+normativeData["Ankle.Power"]["sd"][:,2], facecolor="green", alpha=0.5,linewidth=0)

            pp = PdfPages(str(path+ "consistencyKinetics"+pdfNamePlus+".pdf"))

            pp.savefig(figConsistencyKinetics)
            pp.close()
