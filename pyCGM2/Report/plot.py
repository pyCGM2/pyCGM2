# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

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

try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2.EMG import normalActivation


# ---- convenient plot functions
def temporalPlot(figAxis,acq,
                pointLabel,axis,pointLabelSuffix=None,color=None,linewidth=None,
                title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                customLimits=None):

    '''Plot temporal traces from an acquisition

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         acq (btk.Acquisition): an acquisition
         pointLabel (str): point label
         axis (int): column index of the point values
         pointLabelSuffix (type,Optional[None]): suffix added to the point label
         color (type,Optional[None]): line color
         linewidth (type,Optional[None]): line width
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

    '''

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


def descriptivePlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot descriptive (average and sd corridor) time-normalized traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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

def consistencyPlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot all time-normalized traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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


def meanPlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot the  average time-normalized traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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


def gaitDescriptivePlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot descriptive (average and sd corridor) gait traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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

def gaitConsistencyPlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot all gait traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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

def gaitMeanPlot(figAxis,analysisStructureItem,
                        pointLabel,contextPointLabel,axis,
                        color=None,
                        title=None, xlabel=None, ylabel=None,ylim=None,legendLabel=None,
                        customLimits=None):

    '''Plot average gait traces from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         pointLabel (str): point label
         contextPointLabel (str): event context
         axis (int): column index of the point values
         color (type,Optional[None]): line color
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         ylabel (type,Optional[None]): y-axis label
         ylim (list,Optional[None]): y boundaries
         legendLabel (type,Optional[None]): legend
         customLimits (list,Optional[None]): horizontal lines

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

def stpHorizontalHistogram(figAxis,analysisStructureItem,
                        stpLabel,
                        overall= False,
                        title=None, xlabel=None,xlim=None):
    '''Plot spatio-temporal parameters as histogram from an attribute of an `analysis` instance

    Args:
         figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
         analysisStructureItem (pyCGM2.Processing.analysis.Analysis.(attribute)): an attribute of an `analysis` instance
         stpLabel (str): spatio-temporal label
         title (type,Optional[None]): plot title
         xlabel (type,Optional[None]): x-axis label
         xlim (list,Optional[None]): x boundaries

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


def addNormalActivationLayer(figAxis,normalActivationLabel,fo):
    """display normal muscle activation in the background of a time-normalized trace

    Args:
        figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
        normalActivationLabel (str): muscle label
        fo (int): time-normalized foot off frame

    """

    pos,burstDuration=normalActivation.getNormalBurstActivity(normalActivationLabel,fo)
    for j in range(0,len(pos)):
        figAxis.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],figAxis.get_ylim()[1] , color='g',alpha=0.1))



def addTemporalNormalActivationLayer(figAxis,acq,normalActivationLabel,context):
    """display normal muscle activation in the background of a temporal trace

    Args:
        figAxis (matplotlib.pyplot.Axis): a matplotlib figure axis
        acq (btk.Acquisition): an acquisition
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



def addRectanglePatches(figAxis,clusters,heightProportion = 0.05):
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
