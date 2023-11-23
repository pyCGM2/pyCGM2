"""
Module contains low-level plot functions
"""
from typing import Optional,Union,List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyCGM2.Processing import cycle

from pyCGM2.Tools import btkTools
from pyCGM2.Report import plotUtils

import pyCGM2; LOGGER = pyCGM2.LOGGER

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from pyCGM2.EMG import normalActivation
from pyCGM2.Processing.analysis import AnalysisStructure

# ---- convenient plot functions
def temporalPlot(figAxis:plt.Axes,acq:btk.btkAcquisition,
                pointLabel:str,axis:int,
                pointLabelSuffix:Optional[str]=None,color:Optional[str]=None,linewidth:Optional[str]=None,
                title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None,
                ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                customLimits:Optional[list]=None):
    """Plot temporal traces from an acquisition

    Args:
        figAxis (plt.Axes): a matplotlib figure axes
        acq (btk.btkAcquisition): a btk acquisition
        pointLabel (str): point label
        axis (int): column index of the point values
        pointLabelSuffix (Optional[str], optional): suffix added to the point label. Defaults to None.
        color (Optional[str], optional): line color. Defaults to None.
        linewidth (Optional[str], optional): line width. Defaults to None.
        title (Optional[str], optional): title Defaults to None.
        xlabel (Optional[str], optional): x-axis label. Defaults to None.
        ylabel (Optional[str], optional): y-axis label. Defaults to None.
        ylim (Optional[list], optional): y boundaries. Defaults to None.
        legendLabel (Optional[str], optional): legend. Defaults to None.
        customLimits (Optional[list], optional): horizontal lines. Defaults to None.
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
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None,
                    nan_to_num:bool=True):

    '''Plot descriptive (average and sd corridor) time-normalized traces from an attribute of an `analysis` instance

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
         nan_to_num (bool,optional): nan conversion
                 
    '''


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
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None,
                    nan_to_num:bool=True):

    '''Plot all time-normalized traces from an `analysis` instance

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
         nan_to_num (bool,optional): nan conversion
         
    '''


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
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None,
                    nan_to_num:bool=True):

    '''Plot the  average time-normalized traces from an attribute of an `analysis` instance

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
         nan_to_num (bool,optional): nan conversion

    '''



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
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None):

    '''Plot descriptive (average and sd corridor) gait traces from an attribute of an `analysis` instance

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
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None):

    '''Plot all gait traces from an attribute of an `analysis` instance

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
                    title:Optional[str]=None, xlabel:Optional[str]=None, 
                    ylabel:Optional[str]=None,ylim:Optional[list]=None,legendLabel:Optional[str]=None,
                    customLimits:Optional[list]=None):

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
        lines= figAxis.plot(np.linspace(0,100,101), mean, color=color,linestyle="-")

        stance = analysisStructureItem.pst['stancePhase', contextPointLabel]["mean"]
        double1 = analysisStructureItem.pst['doubleStance1', contextPointLabel]["mean"]
        double2 = analysisStructureItem.pst['doubleStance2', contextPointLabel]["mean"]
        figAxis.axvline(stance,color=color,ls='dashed')
        figAxis.axvline(double1,ymin=0.9, ymax=1.0,color=color,ls='dotted')
        figAxis.axvline(stance-double2,ymin=0.9, ymax=1.0,color=color,ls='dotted')

        if customLimits is not None:
            for value in customLimits:
                figAxis.axhline(value,color=color,ls='dashed')

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
                        xlim:Optional[list]=None):
    '''Plot spatio-temporal parameters as histogram from an attribute of an `analysis` instance

    Args:
         figAxis (plt.Axes): a matplotlib figure axis
         analysisStructureItem (AnalysisStructure): an AnalysisStructure on an analysis instance 
         stpLabel (str): spatio-temporal label
         title (Optional[str], optional): plot title
         xlabel (Optional[str], optional): x-axis label. Defaults to None.
         xlim (Optional[list], optional): x boundaries. Defaults to None.
    '''


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


def addNormalActivationLayer(figAxis:plt.Axes,normalActivationLabel:str,fo:int):
    """display normal muscle activation in the background of a time-normalized trace

    Args:
        figAxis (plt.Axes): a matplotlib figure axis
        normalActivationLabel (str): muscle label
        fo (int): time-normalized foot off frame

    """

    pos,burstDuration=normalActivation.getNormalBurstActivity(normalActivationLabel,fo)
    for j in range(0,len(pos)):
        figAxis.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],figAxis.get_ylim()[1] , color='g',alpha=0.1))



def addTemporalNormalActivationLayer(figAxis:plt.Axes,acq:btk.btkAcquisition,
                                     normalActivationLabel:str,context:str):
    """display normal muscle activation in the background of a temporal trace

    Args:
        figAxis (plt.Axes): a matplotlib figure axis
        acq (btk.btkAcquisition): a Btk acquisition
        normalActivationLabel (str): muscle label
        context (str): event context
    """

    if normalActivationLabel:
        gaitCycles = cycle.construcGaitCycle(acq)

        for cycleIt  in gaitCycles:
            if cycleIt.context == context:
                pos,burstDuration=normalActivation.getNormalBurstActivity_fromCycles(normalActivationLabel,cycleIt.firstFrame,cycleIt.begin, cycleIt.m_contraFO, cycleIt.end, cycleIt.appf)
                for j in range(0,len(pos)):
                    figAxis.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],figAxis.get_ylim()[1] , color='g',alpha=0.1))



def addRectanglePatches(figAxis:plt.Axes,clusters:list,heightProportion:float = 0.05):
    """displat a rectangle

    Args:
        figAxis (plt.Axes): a matplotlib figure axis
        clusters (list): clusters with begin and end indexes
        heightProportion (float, optional): proportion with height. Defaults to 0.05.
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
