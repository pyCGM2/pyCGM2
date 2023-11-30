"""
Module contains low-level plot functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyCGM2.Processing import cycle

from pyCGM2.Tools import btkTools
from pyCGM2.Report import plotUtils

import pyCGM2; LOGGER = pyCGM2.LOGGER

import btk

from pyCGM2.EMG import normalActivation
from pyCGM2.Processing.analysis import AnalysisStructure

from typing import List, Tuple, Dict, Optional, Union, Callable

# ---- convenient plot functions
def temporalPlot(figAxis:plt.Axes,acq:btk.btkAcquisition,
                pointLabel:str,axis:int,
                pointLabelSuffix:Optional[str]=None,color:Optional[str]=None,linewidth:Optional[str]=None,
                title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None,
                ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                customLimits:Optional[List]=None):
    """Plots temporal traces from an acquisition.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        acq (btk.btkAcquisition): A btk acquisition.
        pointLabel (str): Point label.
        axis (int): Column index of the point values.
        pointLabelSuffix (Optional[str], optional): Suffix added to the point label. Defaults to None.
        color (Optional[str], optional): Line color. Defaults to None.
        linewidth (Optional[str], optional): Line width. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
    """

    pointLabel = pointLabel + "_" + pointLabelSuffix if pointLabelSuffix is not None else pointLabel


    flag = btkTools.isPointExist(acq,pointLabel)
    if flag:
        point = acq.GetPoint(pointLabel)
        lines=figAxis.plot(point.GetValues()[:,axis], '-', color= color,linewidth=linewidth)
        appf = 1
    else:
        flag = btkTools.isAnalogExist(acq,pointLabel)
        if flag:
            analog = acq.GetAnalog(pointLabel)
            lines=figAxis.plot(analog.GetValues()[:,axis], '-', color= color,linewidth=linewidth)
            appf = acq.GetNumberAnalogSamplePerFrame()


    if legendLabel is not None  and flag: lines[0].set_label(legendLabel)
    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None: figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None: figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None: figAxis.set_ylim(ylim)

    if flag:
        for ev in btk.Iterate( acq.GetEvents()):
            colorContext = plotUtils.colorContext(ev.GetContext())
            if ev.GetLabel() == "Foot Strike":
                figAxis.axvline( x= (ev.GetFrame()-acq.GetFirstFrame())*appf, color = colorContext, linestyle = "-")
            if ev.GetLabel() == "Foot Off":
                figAxis.axvline( x= (ev.GetFrame()-acq.GetFirstFrame())*appf, color = colorContext, linestyle = "--")


def descriptivePlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None,
                    nan_to_num:bool=True):

    """Plots descriptive (average and sd corridor) time-normalized traces from an analysis instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An attribute of an analysis instance.
        pointLabel (str): Point label.
        contextPointLabel (str): Event context.
        axis (int): Column index of the point values.
        color (Optional[str], optional): Line color. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
        nan_to_num (bool, optional): Convert NaN to number. Defaults to True.
    """


    # check if [ pointlabel , context ] in keys of analysisStructureItem
    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"] != [] else False


    # plot
    if flag:
        mean=analysisStructureItem.data[pointLabel,contextPointLabel]["mean"][:,axis]
        std=analysisStructureItem.data[pointLabel,contextPointLabel]["std"][:,axis]

        if nan_to_num:
            mean = np.nan_to_num(mean)
            std = np.nan_to_num(std)


        line= figAxis.plot(np.linspace(0,100,101), mean, color=color,linestyle="-")
        figAxis.fill_between(np.linspace(0,100,101), mean-std, mean+std, facecolor=color, alpha=0.5,linewidth=0)

        if customLimits is not None:
            for value in customLimits:
                figAxis.axhline(value,color=color,ls='dashed')
       
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")
        line= figAxis.plot(np.linspace(0,100,101), np.zeros(101), color=color,linestyle="-")        

    if legendLabel is not None: line[0].set_label(legendLabel)

        
    
    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None:figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None:figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None:figAxis.set_ylim(ylim)

def consistencyPlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None,
                    nan_to_num:bool=True):

    """Plots all time-normalized traces from an `analysis` instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An attribute of an `analysis` instance.
        pointLabel (str): Point label.
        contextPointLabel (str): Event context.
        axis (int): Column index of the point values.
        color (Optional[str], optional): Line color. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
        nan_to_num (bool, optional): Convert NaN to number. Defaults to True.
    """


    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            n = len(analysisStructureItem.data[pointLabel,contextPointLabel]["values"])
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"] != [] else False

    # plot
    if flag:
        values= np.zeros((101,n))
        i=0
        for val in analysisStructureItem.data[pointLabel,contextPointLabel]["values"]:
           values[:,i] = val[:,axis]
           i+=1

        lines = figAxis.plot(np.linspace(0,100,101), values, color=color)

        if customLimits is not None:
           for value in customLimits:
               figAxis.axhline(value,color=color,ls='dashed')
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")


    if legendLabel is not None and flag: lines[0].set_label(legendLabel)

    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None:figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None:figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None:figAxis.set_ylim(ylim)


def meanPlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None,
                    nan_to_num:bool=True):

    """Plots the average time-normalized traces from an attribute of an `analysis` instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An attribute of an `analysis` instance.
        pointLabel (str): Point label.
        contextPointLabel (str): Event context.
        axis (int): Column index of the point values.
        color (Optional[str], optional): Line color. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
        nan_to_num (bool, optional): Convert NaN to number. Defaults to True.
    """



    # check if [ pointlabel , context ] in keys of analysisStructureItem
    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"] != [] else False


    # plot
    if flag:
        mean=analysisStructureItem.data[pointLabel,contextPointLabel]["mean"][:,axis]

        if nan_to_num:
            mean = np.nan_to_num(mean)

        lines= figAxis.plot(np.linspace(0,100,101), mean, color=color,linestyle="-")

        if customLimits is not None:
            for value in customLimits:
                figAxis.axhline(value,color=color,ls='dashed')
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")


    if legendLabel is not None  and flag: lines[0].set_label(legendLabel)
    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None: figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None: figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None: figAxis.set_ylim(ylim)


def gaitDescriptivePlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None):

    """Plots descriptive (average and sd corridor) gait traces from an attribute of an `analysis` instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An attribute of an `analysis` instance.
        pointLabel (str): Point label.
        contextPointLabel (str): Event context.
        axis (int): Column index of the point values.
        color (Optional[str], optional): Line color. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
    """



    # check if [ pointlabel , context ] in keys of analysisStructureItem
    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"]!=[]  else False

    # plot
    if flag:
        mean=analysisStructureItem.data[pointLabel,contextPointLabel]["mean"][:,axis]
        std=analysisStructureItem.data[pointLabel,contextPointLabel]["std"][:,axis]


        line= figAxis.plot(np.linspace(0,100,101), mean, color=color,linestyle="-")
        figAxis.fill_between(np.linspace(0,100,101), mean-std, mean+std, facecolor=color, alpha=0.5,linewidth=0)

        # add gait phases
        stance = analysisStructureItem.pst['stancePhase', contextPointLabel]["mean"]
        double1 = analysisStructureItem.pst['doubleStance1', contextPointLabel]["mean"]
        double2 = analysisStructureItem.pst['doubleStance2', contextPointLabel]["mean"]
        figAxis.axvline(stance,color=color,ls='dashed')
        figAxis.axvline(double1,ymin=0.9, ymax=1.0,color=color,ls='dotted')
        figAxis.axvline(stance-double2,ymin=0.9, ymax=1.0,color=color,ls='dotted')

        if customLimits is not None:
            for value in customLimits:
                figAxis.axhline(value,color=color,ls='dashed')
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")

    if legendLabel is not None: line[0].set_label(legendLabel)
    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None:figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None:figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None:figAxis.set_ylim(ylim)

def gaitConsistencyPlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None):

    """Plots all gait traces from an attribute of an `analysis` instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An attribute of an `analysis` instance.
        pointLabel (str): Point label.
        contextPointLabel (str): Event context.
        axis (int): Column index of the point values.
        color (Optional[str], optional): Line color. Defaults to None.
        title (Optional[str], optional): Title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        ylim (Optional[List], optional): Y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): Legend. Defaults to None.
        customLimits (Optional[List], optional): Horizontal lines. Defaults to None.
    """

    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            n = len(analysisStructureItem.data[pointLabel,contextPointLabel]["values"])
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"]!=[] else False

    # plot
    if flag:
        values= np.zeros((101,n))
        i=0
        for val in analysisStructureItem.data[pointLabel,contextPointLabel]["values"]:

            values[:,i] = val[:,axis]

            i+=1

        lines = figAxis.plot(np.linspace(0,100,101), values, color=color)

        for valStance,valDouble1,valDouble2, in zip(analysisStructureItem.pst['stancePhase', contextPointLabel]["values"],
                                                    analysisStructureItem.pst['doubleStance1', contextPointLabel]["values"],
                                                    analysisStructureItem.pst['doubleStance2', contextPointLabel]["values"]):

            figAxis.axvline(valStance,color=color,ls='dashed')
            figAxis.axvline(valDouble1,ymin=0.9, ymax =1.0 ,color=color,ls='dotted')
            figAxis.axvline(valStance-valDouble2,ymin=0.9, ymax =1.0 ,color=color,ls='dotted')

        if customLimits is not None:
           for value in customLimits:
               figAxis.axhline(value,color=color,ls='dashed')
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")


    if legendLabel is not None and flag: lines[0].set_label(legendLabel)

    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None:figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None:figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None:figAxis.set_ylim(ylim)

def gaitMeanPlot(figAxis:plt.Axes,
                    analysisStructureItem:AnalysisStructure,
                    pointLabel:str,contextPointLabel:str,axis:int,
                    color:Optional[str]=None,
                    alpha:Optional[float]=None,
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[List]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[List]=None):

    '''Plot average traces from an attribute of an `analysis` instance

    Args:
        figAxis (plt.Axes): a matplotlib figure axis
        analysisStructureItem (AnalysisStructure): an attribute of an `analysis` instance
        pointLabel (str): point label
        contextPointLabel (str): event context
        axis (int): column index of the point values
        color (Optional[str], optional): line color. Defaults to None.
        title (Optional[str], optional): title Defaults to None.
        xlabel (Optional[str], optional): x-axis label. Defaults to None.
        ylabel (Optional[str], optional): y-axis label. Defaults to None.
        ylim (Optional[list], optional): y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): legend. Defaults to None.
        customLimits (Optional[list], optional): horizontal lines. Defaults to None.

    '''



    # check if [ pointlabel , context ] in keys of analysisStructureItem
    flag = False
    for key in analysisStructureItem.data.keys():
        if key[0] == pointLabel and key[1] == contextPointLabel:
            flag = True if analysisStructureItem.data[pointLabel,contextPointLabel]["values"] != [] else False


    # plot
    if flag:
        mean=analysisStructureItem.data[pointLabel,contextPointLabel]["mean"][:,axis]
        lines= figAxis.plot(np.linspace(0,100,101), mean, color=color,linestyle="-",alpha=alpha)

        stance = analysisStructureItem.pst['stancePhase', contextPointLabel]["mean"]
        double1 = analysisStructureItem.pst['doubleStance1', contextPointLabel]["mean"]
        double2 = analysisStructureItem.pst['doubleStance2', contextPointLabel]["mean"]
        figAxis.axvline(stance,color=color,ls='dashed')
        figAxis.axvline(double1,ymin=0.9, ymax=1.0,color=color,ls='dotted')
        figAxis.axvline(stance-double2,ymin=0.9, ymax=1.0,color=color,ls='dotted')

        if customLimits is not None:
            for value in customLimits:
                figAxis.axhline(value,color=color,ls='dashed')
    else:
        LOGGER.logger.warning(f"[pyCGM2] - no values to diplay. The label {pointLabel}-{contextPointLabel} is not within your analysis instance ")

    if legendLabel is not None  and flag: lines[0].set_label(legendLabel)
    if title is not None: figAxis.set_title(title ,size=8)
    figAxis.set_xlim([0.0,100])
    figAxis.tick_params(axis='x', which='major', labelsize=6)
    figAxis.tick_params(axis='y', which='major', labelsize=6)
    if xlabel is not None: figAxis.set_xlabel(xlabel,size=8)
    if ylabel is not None: figAxis.set_ylabel(ylabel,size=8)
    if ylim is not None: figAxis.set_ylim(ylim)

def stpHorizontalHistogram(figAxis:plt.Axes,analysisStructureItem:AnalysisStructure,
                        stpLabel:str,
                        overall:bool= False,
                        title:Optional[str]=None, 
                        xlabel:Optional[str]=None,
                        xlim:Optional[List]=None):
    """Plots spatio-temporal parameters as a histogram from an attribute of an `analysis` instance.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        analysisStructureItem (AnalysisStructure): An AnalysisStructure on an analysis instance.
        stpLabel (str): Spatio-temporal label.
        overall (bool, optional): Plot overall data. Defaults to False.
        title (Optional[str], optional): Plot title. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        xlim (Optional[List], optional): X boundaries. Defaults to None.
    """


    if (stpLabel,"Right") in analysisStructureItem.keys() and (stpLabel,"Left") in analysisStructureItem.keys():
        overallData = np.concatenate([analysisStructureItem[stpLabel,"Left"]["values"],analysisStructureItem[stpLabel,"Right"]["values"]] )
        mean_L = analysisStructureItem[stpLabel,"Left"]["mean"]
        err_L = analysisStructureItem[stpLabel,"Left"]["std"]
        mean_R = analysisStructureItem[stpLabel,"Right"]["mean"]
        err_R = analysisStructureItem[stpLabel,"Right"]["std"]

    if not (stpLabel,"Right") in analysisStructureItem.keys() and (stpLabel,"Left") in analysisStructureItem.keys():
        overallData = analysisStructureItem[stpLabel,"Left"]["values"]
        mean_L = analysisStructureItem[stpLabel,"Left"]["mean"]
        err_L = analysisStructureItem[stpLabel,"Left"]["std"]
        mean_R = 0
        err_R = 0

    if  (stpLabel,"Right") in analysisStructureItem.keys() and not (stpLabel,"Left") in analysisStructureItem.keys():
        overallData = analysisStructureItem[stpLabel,"Right"]["values"]
        mean_L = 0
        err_L = 0
        mean_R = analysisStructureItem[stpLabel,"Right"]["mean"]
        err_R = analysisStructureItem[stpLabel,"Right"]["std"]

    if overall:
        mean = np.mean(overallData)
        err = np.std(overallData)
        figAxis.barh([0], [ mean], color='purple', xerr=[err])
        figAxis.set_ylabel("Overall",size=8)
        figAxis.set_yticklabels( [""])
        figAxis.text(0,0,round(mean,2))

    else:
        figAxis.barh([0,1], [mean_L,mean_R],  xerr=[err_L,err_R], color=["red","blue"])
        figAxis.set_yticks([0,1])
        figAxis.set_yticklabels(["L","R"],size=8)
        figAxis.text(0,0,round(mean_L,2))
        figAxis.text(0,1,round(mean_R,2))

    if title is not None: figAxis.set_title(title ,size=8)
    if xlabel is not None: figAxis.set_xlabel(xlabel,size=8)
    if xlim is not None: figAxis.set_xlim(xlim)

    figAxis.tick_params(axis='x', which='major', labelsize=6)


def addNormalActivationLayer(figAxis:plt.Axes,normalActivationLabel:str,fo:int,color:str="g",edgecolor:str="red",alpha:float=0.1,position:Optional[str]=None):
    """Displays normal muscle activation in the background of a time-normalized trace.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        normalActivationLabel (str): Muscle label.
        fo (int): Time-normalized foot off frame.
        color(str): color line
        edgecolor(str): edge color
        alpha(float): transparency value
        position(Optional[str]): position of the rectangle patch ( None, Upper or Lower)
    """
    
    prop = 0.1*(figAxis.get_ylim()[1]-figAxis.get_ylim()[0])
    figAxis.set_ylim(bottom=figAxis.get_ylim()[0]-prop, top=figAxis.get_ylim()[1]+prop)

    pos,burstDuration=normalActivation.getNormalBurstActivity(normalActivationLabel,fo)
    for j in range(0,len(pos)):
        if position is None:
            figAxis.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],figAxis.get_ylim()[1] , facecolor=color,alpha=alpha))
        if position=="Upper":
            figAxis.add_patch(plt.Rectangle((pos[j],figAxis.get_ylim()[1]-prop),burstDuration[j],figAxis.get_ylim()[1] , facecolor=color,alpha=alpha,edgecolor=edgecolor))
        if position=="Lower":
            figAxis.add_patch(plt.Rectangle((pos[j],figAxis.get_ylim()[0]+prop),burstDuration[j],figAxis.get_ylim()[0] , facecolor=color,alpha=alpha,edgecolor=edgecolor))


def addTemporalNormalActivationLayer(figAxis:plt.Axes,acq:btk.btkAcquisition,
                                     normalActivationLabel:str,context:str):
    """Displays normal muscle activation in the background of a temporal trace.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        acq (btk.btkAcquisition): A Btk acquisition.
        normalActivationLabel (str): Muscle label.
        context (str): Event context.
    """
    if normalActivationLabel:
        gaitCycles = cycle.construcGaitCycle(acq)

        for cycleIt  in gaitCycles:
            if cycleIt.context == context:
                pos,burstDuration=normalActivation.getNormalBurstActivity_fromCycles(normalActivationLabel,cycleIt.firstFrame,cycleIt.begin, cycleIt.m_contraFO, cycleIt.end, cycleIt.appf)
                for j in range(0,len(pos)):
                    figAxis.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],figAxis.get_ylim()[1] , color='g',alpha=0.1))
                    



def addRectanglePatches(figAxis:plt.Axes,clusters:List,heightProportion:float = 0.05):
    """Displays a rectangle.

    Args:
        figAxis (plt.Axes): A matplotlib figure axis.
        clusters (List): Clusters with begin and end indexes.
        heightProportion (float, optional): Proportion with height. Defaults to 0.05.
    """
    ymax = figAxis.get_ylim()[1]
    ymin = figAxis.get_ylim()[0]
    height = (ymax-ymin)*heightProportion
    for cluster in clusters:
        begin = cluster[0]
        end = cluster[1]
        rectanglePatch = mpatches.Rectangle((begin, ymax-height),
                                        end-begin,
                                        height,
                                        color="k")

        figAxis.add_patch(rectanglePatch)
