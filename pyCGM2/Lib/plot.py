"""
This module gathers convenient functions for plotting Kinematics, Kinetics and EMG.
All functions return a matplotlib figure instance

"""

import matplotlib.pyplot as plt
import numpy as np

import pyCGM2
from pyCGM2.Report.Viewers import plotViewers
from pyCGM2.Report.Viewers import emgPlotViewers
from pyCGM2.Report.Viewers import customPlotViewers
from pyCGM2.Report.Viewers import  comparisonPlotViewers
from pyCGM2.Report import plot
from pyCGM2.Report import plotFilters
from pyCGM2.Report import normativeDatasets
from pyCGM2.Processing.Scores import scoreFilters
from pyCGM2.Processing.Scores import scoreProcedures
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.EMG import emgManager
from pyCGM2.Report.Viewers import musclePlotViewers
from pyCGM2.Utils import files
from pyCGM2.Processing.Classification import classificationFilters, classificationProcedures
from pyCGM2.Report.Viewers import groundReactionPlotViewers
from pyCGM2.Processing.analysis import Analysis
from pyCGM2.Report.normativeDatasets import NormativeData

from typing import List, Tuple, Dict, Optional,Union


def plotTemporalKinematic(DATA_PATH:str, modelledFilename:str,bodyPart:str, pointLabelSuffix:Optional[str]=None,
                          exportPdf:bool=False,OUT_PATH:Optional[str] = None, outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
                          autoYlim:bool=False,
                          **kwargs):
    """
    Displays temporal traces of CGM kinematic outputs for a specified body part.

    This function generates and optionally exports a plot showing the kinematic outputs of a 
    specified body part over time. The data is read from a C3D file, and the plot can be customized 
    in various ways.

    Args:
        DATA_PATH (str): Path to the data directory.
        modelledFilename (str): Filename of the C3D file including kinematic output.
        bodyPart (str): Body part to plot (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        pointLabelSuffix (Optional[str]): Suffix added to model outputs. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        btkAcq (Optional[btk.Acquisition]): If provided, uses this acquisition instead of loading from `modelledFilename`.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotTemporalKinematic("/myDATA/", "file1.c3d", "LowerLimb")

    Note:
        Ensure the specified `bodyPart` is one of the valid choices. Invalid input will raise an exception.
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")

    if exportPdf or exportPng:
        if outputName is None:
            filenameOut =  modelledFilename+"-Temporal Kinematics ["+ bodyPart.name+"]"
        else:
            filenameOut =  outputName+"-Temporal Kinematics ["+ bodyPart.name+"]"

    if "btkAcq" in kwargs.keys() and  kwargs["btkAcq"] is not None:
        acq = kwargs["btkAcq"]
        btkTools.sortedEvents(acq)
    else:
        acq =btkTools.smartReader(DATA_PATH + modelledFilename)

    kv = plotViewers.TemporalKinematicsPlotViewer(acq,pointLabelSuffix=pointLabelSuffix,bodyPart = bodyPart)
    kv.setAutomaticYlimits(autoYlim)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Temporal Kinematics ["+ bodyPart.name+"]")
    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plotTemporalKinetic(DATA_PATH:str, modelledFilenames:str,bodyPart:str,
                        pointLabelSuffix:Optional[str]=None,exportPdf:bool=False,
                        OUT_PATH:Optional[str]= None, outputName:Optional[str]=None,show:bool=True,title:str=None,
                        exportPng:bool=False,autoYlim:bool=False,**kwargs):

    """
    Displays temporal traces of CGM kinetic outputs for a specified body part.

    This function generates and optionally exports a plot showing the kinetic outputs of a specified body part over time. 
    The data is read from a C3D file, and the plot can be customized in various ways.

    Args:
        DATA_PATH (str): Path to the data directory.
        modelledFilenames (str): Filename of the C3D file including kinetic output.
        bodyPart (str): Body part to plot (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        pointLabelSuffix (Optional[str]): Suffix added to model outputs. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        btkAcq (Optional[btk.Acquisition]): If provided, uses this acquisition instead of loading from `modelledFilenames`.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotTemporalKinetic("/myDATA/", "file1.c3d", "LowerLimb")

    Note:
        Ensure the specified `bodyPart` is one of the valid choices. Invalid input will raise an exception.
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH


    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")

    if exportPdf or exportPng:
        if outputName is None:
            filenameOut =  modelledFilenames+"-Temporal Kinetics["+ bodyPart.name+"]"
        else:
            filenameOut =  outputName+"-Temporal Kinetics ["+ bodyPart.name+"]"

    if "btkAcq" in kwargs.keys() and  kwargs["btkAcq"] is not None:
        acq = kwargs["btkAcq"]
        btkTools.sortedEvents(acq)

    else:
        acq =btkTools.smartReader(DATA_PATH+modelledFilenames)

    kv = plotViewers.TemporalKineticsPlotViewer(acq,pointLabelSuffix=pointLabelSuffix,bodyPart = bodyPart)
    kv.setAutomaticYlimits(autoYlim)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Temporal Kinetics ["+ bodyPart.name+"]")
    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plotTemporalReaction(DATA_PATH:str, modelledFilenames:str,
                        pointLabelSuffix:Optional[str]=None,exportPdf:bool=False,
                        OUT_PATH:Optional[str]= None, outputName:Optional[str]=None,show:bool=True,title:str=None,
                        exportPng:bool=False,autoYlim:bool=False,**kwargs):

    """
    Displays temporal traces of the ground reaction.



    Args:
        DATA_PATH (str): Path to the data directory.
        modelledFilenames (str): Filename of the C3D file including kinetic output.
        pointLabelSuffix (Optional[str]): Suffix added to model outputs. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        btkAcq (Optional[btk.Acquisition]): If provided, uses this acquisition instead of loading from `modelledFilenames`.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotTemporalKinetic("/myDATA/", "file1.c3d", "LowerLimb")

    Note:
        Ensure the specified `bodyPart` is one of the valid choices. Invalid input will raise an exception.
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if exportPdf or exportPng:
        if outputName is None:
            filenameOut =  modelledFilenames+"-Temporal Ground reaction"
        else:
            filenameOut =  outputName+"-Temporal Ground reaction"

    if "btkAcq" in kwargs.keys() and  kwargs["btkAcq"] is not None:
        acq = kwargs["btkAcq"]
        btkTools.sortedEvents(acq)

    else:
        acq =btkTools.smartReader(DATA_PATH+modelledFilenames)

    kv = groundReactionPlotViewers.TemporalReactionPlotViewer(acq,pointLabelSuffix=pointLabelSuffix)
    kv.setAutomaticYlimits(autoYlim)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Temporal Ground reaction ")
    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plotTemporalEMG(DATA_PATH:str, processedEmgfile:str,
                    rectify:bool = True, exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,
                    ignoreNormalActivity:bool= False,exportPng:bool=False,OUT_PATH:Optional[str]=None,autoYlim:bool=False,
                    **kwargs):
    """
    Displays temporal traces of EMG signals from a processed EMG file.

    This function visualizes EMG data, allowing options for rectification, title modification, 
    and exportation in PDF or PNG formats. It can display rectified or raw EMG signals, and 
    optionally ignores normal activity in the background.

    Args:
        DATA_PATH (str): Path to the data directory.
        processedEmgfile (str): Filename of the C3D file with EMG data.

    Keyword Args:
        rectify (bool): Display rectified (True) or raw (False) signal. Defaults to True.
        exportPdf (bool): Export the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): Show the matplotlib figure. Defaults to True.
        title (Optional[str]): Modify the plot panel title. Defaults to None.
        ignoreNormalActivity (bool): Disable display of normal activity in the background. Defaults to False.
        exportPng (bool): Export the plot as a PNG. Defaults to False.
        OUT_PATH (Optional[str]): Specify an output path different than `DATA_PATH`. Defaults to None.
        autoYlim (bool): Ignore predefined Y-axis boundaries. Defaults to False.

    Additional Keyword Args:
        btkAcq (Optional[btk.Acquisition]): Use this acquisition instead of loading from `processedEmgfile`.
        forceEmgManager (Optional[pyCGM2.EMG.EmgManager]): Use a specific EmgManager instance.

    Returns:
        Union[List[matplotlib.figure.Figure], Tuple[List[matplotlib.figure.Figure], List[str]]]: 
        A list of Matplotlib figure objects. If exporting as PNG, returns a tuple of the list of figures and list of filenames.

    Examples:
        >>> figures = plotTemporalEMG("/myDATA/", "file1.c3d")

    Note:
        The function can generate multiple plots depending on the number of EMG channels. Each plot can be exported separately if required.
    """



    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if "btkAcq" in kwargs.keys() and  kwargs["btkAcq"] is not None:
        acq = kwargs["btkAcq"]
    else:
        acq =btkTools.smartReader(DATA_PATH+processedEmgfile)

    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)
    emgChannels = emg.getChannels()

    emgChannels_list=  [emgChannels[i:i+10] for i in range(0, len(emgChannels), 10)]


    pageNumber = len(emgChannels_list)

    figs=[]
    outfilenames=[]

    exportFlag = True if exportPdf or exportPng else False

    count = 0
    for i in range(0,pageNumber):

        if exportFlag and pageNumber>1:
            if outputName is None:
                filenameOut =  processedEmgfile+"-TemporalEmgPlot"+"[rectify]-"+str(count) if rectify else processedEmgfile+"-TemporalEmgPlot"+"[raw]-"+count
            else:
                filenameOut =  outputName+"-TemporalEmgPlot"+"[rectify]-"+str(count) if rectify else title+"-TemporalEmgPlot"+"[raw]-"+count
        else:
            if outputName is None:
                filenameOut =  processedEmgfile+"-TemporalEmgPlot"+"[rectify]" if rectify else processedEmgfile+"-TemporalEmgPlot"+"[raw]"
            else:
                filenameOut =  outputName+"-TemporalEmgPlot"+"[rectify]" if rectify else title+"-TemporalEmgPlot"+"[raw]"

        # # viewer
        kv = emgPlotViewers.TemporalEmgPlotViewer(acq)
        kv.setEmgManager(emg)
        kv.selectEmgChannels(emgChannels_list[i])
        kv.ignoreNormalActivty(ignoreNormalActivity)
        kv. setEmgRectify(rectify)

        kv.setAutomaticYlimits(autoYlim)

        # # filter

        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        if title is not None:
            if pageNumber>1:
                pf.setTitle(title+"-TemporalEmgPlot"+"[rectify]-"+str(count) if rectify else title+"-TemporalEmgPlot"+"[raw]-"+str(count))
            else:
                pf.setTitle(title+"-TemporalEmgPlot"+"[rectify]" if rectify else title+"-TemporalEmgPlot"+"[raw]")
        if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
        fig = pf.plot()

        if exportPng:
            fig.savefig(OUT_PATH+filenameOut+".png")

        if exportPng:
            outfilenames.append(filenameOut+".png")

        figs.append(fig)

        count+=1
    if show: plt.show()

    if exportPng:
        return figs,outfilenames
    else:
        return figs



def plotDescriptiveEnvelopEMGpanel(DATA_PATH:str,analysis:Analysis,
                                normalized:bool=False, eventType:str="Gait",exportPdf:bool=False,
                                OUT_PATH:Optional[str]= None,outputName=None,
                                show:bool=True,title=None,exportPng=False,autoYlim:bool=False,**kwargs):
    """
    Displays average and standard deviation of time-normalized EMG envelopes.

    This function visualizes descriptive statistics (average and standard deviation) of EMG envelopes 
    for a given analysis instance. It offers options for amplitude normalization and event type specification.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing EMG data.
        normalized (bool): If True, displays normalized EMG amplitude. Defaults to False.
        eventType (str): Event type to consider ('Gait' or other). Defaults to 'Gait'.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        forceEmgManager (Optional[pyCGM2.EMG.EmgManager]): Use a specific EmgManager instance.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotDescriptiveEnvelopEMGpanel("/myDATA/", analysisInstance)

    Note:
        The function allows for detailed visualization of EMG data, which can be tailored based on analysis needs. 
        'Gait' event type assumes cycle definition from foot strike and searches for foot off events.
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH


    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)
    emgChannels = emg.getChannels()

    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-DescriptiveEmgEnv"+"[No Normalized]-" if not normalized else outputName+"-DescriptiveEmgEnv"+"[Normalized]"

    # viewer

    kv = emgPlotViewers.EnvEmgGaitPlotPanelViewer(analysis)
    kv.setEmgManager(emg)
    kv.selectEmgChannels(emgChannels)
    kv.setNormalizedEmgFlag(normalized)

    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)

    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)

    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plotConsistencyEnvelopEMGpanel(DATA_PATH:str,analysis:Analysis, normalized:bool=False,eventType:str="Gait",exportPdf:bool=False,
    OUT_PATH=None, outputName=None,show:bool=True,title=None,exportPng=False,autoYlim:bool=False,**kwargs):

    """
    Displays all-cycle time-normalized EMG envelopes from an analysis instance.

    This function visualizes EMG data for all cycles, highlighting the consistency across repetitions. 
    It allows options for amplitude normalization, event type specification, and exportation in various formats.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing EMG data.
        normalized (bool): If True, displays normalized EMG amplitude. Defaults to False.
        eventType (str): Event type to consider ('Gait' or other). Defaults to 'Gait'.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        forceEmgManager (Optional[pyCGM2.EMG.EmgManager]): Use a specific EmgManager instance.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotConsistencyEnvelopEMGpanel("/myDATA/", analysisInstance)

    Note:
        The function is particularly useful for analyzing the consistency of EMG patterns across multiple gait cycles 
        or other repetitive movements. The 'Gait' event type assumes cycle definition from foot strike and searches for foot off events.
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)

    emgChannels = emg.getChannels()

    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-ConsistencyEmgEnv"+"[No Normalized]-" if not normalized else outputName+"-DescriptiveEmgEnv"+"[Normalized]"

    # viewer
    kv = emgPlotViewers.EnvEmgGaitPlotPanelViewer(analysis)
    kv.setEmgManager(emg)
    kv.selectEmgChannels(emgChannels)
    kv.setNormalizedEmgFlag(normalized)

    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)

    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle( title+"-ConsistencyEmgEnv"+"[No Normalized]-" if not normalized else title+"-DescriptiveEmgEnv"+"[Normalized]")
    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plot_spatioTemporal(DATA_PATH:str,analysis:Analysis,
        exportPdf:bool=False,
        OUT_PATH:Optional[str]=None,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,autoYlim:bool=False):
    """
    Displays spatio-temporal parameters as horizontal histograms.

    This function visualizes spatio-temporal parameters from a given analysis instance, 
    showing them in a format of horizontal histograms. It allows for exportation in PDF or PNG 
    formats and offers options for title modification and automatic Y-axis limit adjustment.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing spatio-temporal data.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, displays the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_spatioTemporal("/myDATA/", analysisInstance)

    Note:
        The function is particularly useful for analyzing and visualizing spatio-temporal gait parameters 
        in a concise and intuitive format.
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:  outputName = "pyCGM2-analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-SpatioTemporal parameters"

    stpv = plotViewers.SpatioTemporalPlotViewer(analysis)
    stpv.setNormativeDataset(normativeDatasets.NormalSTP())

    stpv.setAutomaticYlimits(autoYlim)
    # filter
    stppf = plotFilters.PlottingFilter()
    stppf.setViewer(stpv)
    if title is not None: stppf.setTitle(title+"-SpatioTemporal parameters")
    if exportPdf: stppf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = stppf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_DescriptiveKinematic(DATA_PATH:str,analysis:Analysis,bodyPart:str,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False):
    """
    Displays average and standard deviation of time-normalized kinematic outputs.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing kinematic data.
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_DescriptiveKinematic("/data/path", analysisInstance, "LowerLimb", normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")


    if outputName is None:
        outputName = "pyCGM2-analysis ["+ bodyPart.name+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-descriptive Kinematics ["+ bodyPart.name+"]"


    # filter 1 - descriptive kinematic panel
    #-------------------------------------------
    # viewer

    kv = plotViewers.NormalizedKinematicsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix,bodyPart=bodyPart)

    
    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)


    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-descriptive Kinematics ["+ bodyPart.name+"]")
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plot_ConsistencyKinematic(DATA_PATH:str,analysis:Analysis,bodyPart:str,normativeDataset:NormativeData,
                              pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
                              OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
                              autoYlim:bool=False):

    """
    Displays time-normalized kinematic outputs across cycles.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing kinematic data.
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_ConsistencyKinematic("/data/path", analysisInstance, "LowerLimb", normativeDataset)
    """
    
    
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")


    if outputName is None:
        outputName = "PyCGM2-Analysis ["+ bodyPart.name+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-consistency Kinematics ["+ bodyPart.name+"]"


    kv = plotViewers.NormalizedKinematicsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix,bodyPart=bodyPart)

    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-consistency  Kinematics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_DescriptiveKinetic(DATA_PATH:str,analysis:Analysis,bodyPart:str,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,
        exportPng:bool=False,autoYlim:bool=False):
    """
    Displays average and standard deviation of time-normalized kinetic outputs.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing kinematic data.
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_DescriptiveKinetic("/data/path", analysisInstance, "LowerLimb", normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")


    if outputName is None:
        outputName = "PyCGM2-Analysis ["+ bodyPart.name+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-descriptive Kinetics ["+ bodyPart.name+"]"

    kv = plotViewers.NormalizedKineticsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix,bodyPart=bodyPart)

    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)



    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-descriptive  Kinetics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_ConsistencyKinetic(DATA_PATH:str,analysis:Analysis,bodyPart:str, normativeDataset:NormativeData,
                            pointLabelSuffix:Optional[str]=None,eventType:str="Gait",OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,
                            title:Optional[str]=None,exportPng:bool=False,autoYlim:bool=False):
    """
    Displays time-normalized kinetic outputs across cycles.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing kinematic data.
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_ConsistencyKinetic("/data/path", analysisInstance, "LowerLimb", normativeDataset)
    """


    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")

    if outputName is None:
        outputName = "PyCGM2-Analysis ["+ bodyPart.name+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-consistency Kinetics ["+ bodyPart.name+"]"

    kv = plotViewers.NormalizedKineticsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix,bodyPart=bodyPart)

    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-consistency  Kinetics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_MAP(DATA_PATH:str,analysis:Analysis,normativeDataset:NormativeData,
            exportPdf:bool=False,outputName:Optional[str]=None,pointLabelSuffix:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,OUT_PATH:Optional[str]=None,
            autoYlim:bool=False):
    """
    Displays a histogram of the Movement Analysis Profile (MAP).

    This function visualizes the MAP, which represents a comparison of an individual's movement data to normative 
    datasets. It is useful for a comprehensive overview of the movement profile across different aspects.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing movement data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_MAP("/data/path", analysisInstance, normativeDataset)
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-Map"

    #compute
    gps =scoreProcedures.CGM1_GPS(pointSuffix=pointLabelSuffix)
    scf = scoreFilters.ScoreFilter(gps,analysis, normativeDataset)
    scf.compute()

    #plot
    kv = plotViewers.GpsMapPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    kv.setAutomaticYlimits(autoYlim)

    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-Map")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareKinematic(DATA_PATH:str,analyses:List[Analysis],legends:List,context:List,bodyPart:List,normativeDataset:NormativeData,
                    plotType="Descriptive",eventType:str="Gait",pointSuffixes=None,show:bool=True,title:Optional[str]=None,
                    OUT_PATH = None,outputName:Optional[str]=None,exportPng:bool=False,exportPdf:bool=False,autoYlim:bool=False):
    """
    Plots kinematics from different analysis instances for comparison.

    This function visualizes and compares kinematic data from multiple analysis instances. It supports 
    descriptive and consistency plot types and can compare data across different contexts and body parts.

    Args:
        DATA_PATH (str): Path to the data directory.
        analyses (List[Analysis]): List of Analysis instances to compare.
        legends (List[str]): Labels representing each analysis instance.
        context (str): Context of the event (e.g., 'Left', 'Right').
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): Normative data instance for comparison.
        plotType (str): Type of plot ('Descriptive' or 'Consistency'). Defaults to 'Descriptive'.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        pointSuffixes (Optional[List[str]]): Suffixes previously added to model outputs. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = compareKinematic("/data/path", [analysis1, analysis2], ["pre", "post"], "Left", "LowerLimb", normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "pyCGM2-Comparison"+"-"+context+" ["+ bodyPart+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-"+context+" ["+ bodyPart+"]"+"- Kinematics Comparison"

    i=1
    for analysis in analyses:
        if analysis.kinematicStats.data == {}:
            raise Exception("[pyCGM2]: Kinetic comparison aborted. Analysis [%i] has no kinematic data"%(i))
        i+=1


    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")

    kv = comparisonPlotViewers.KinematicsPlotComparisonViewer(analyses,context,legends,bodyPart=bodyPart,pointLabelSuffix_lst=pointSuffixes)

    kv.setAutomaticYlimits(autoYlim)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Kinematic comparison")
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareKinetic(DATA_PATH:str,analyses:List[Analysis],legends:List[str],context:List[str],bodyPart:str,normativeDataset:NormativeData,
    plotType="Descriptive",eventType:str="Gait",pointSuffixes=None,show:bool=True,title:Optional[str]=None,
    OUT_PATH:Optional[str]=None,outputName:Optional[str]=None,exportPng:bool=False,exportPdf:bool=False,autoYlim:bool=False):

    """
    Plots kinetics from different analysis instances for comparison.

    This function visualizes and compares kinetic data from multiple analysis instances. It supports 
    descriptive and consistency plot types and can compare data across different contexts and body parts.

    Args:
        DATA_PATH (str): Path to the data directory.
        analyses (List[Analysis]): List of Analysis instances to compare.
        legends (List[str]): Labels representing each analysis instance.
        context (str): Context of the event (e.g., 'Left', 'Right').
        bodyPart (str): Body part to analyze (choices: 'LowerLimb', 'Trunk', 'UpperLimb').
        normativeDataset (NormativeData): Normative data instance for comparison.
        plotType (str): Type of plot ('Descriptive' or 'Consistency'). Defaults to 'Descriptive'.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        pointSuffixes (Optional[List[str]]): Suffixes previously added to model outputs. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = compareKinetic("/data/path", [analysis1, analysis2], ["pre", "post"], "Left", "LowerLimb", normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "pyCGM2-Comparison"+"-"+context+" ["+ bodyPart+"]"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-"+context+" ["+ bodyPart+"]"+"- Kinetics Comparison "


    i=1
    for analysis in analyses:
        if analysis.kineticStats.data == {}:
            raise Exception("[pyCGM2]: Kinetic comparison aborted. Analysis [%i] has no kinetic data"%(i))
        i+=1


    if bodyPart == "LowerLimb":
        bodyPart = enums.BodyPartPlot.LowerLimb
    elif bodyPart == "Trunk":
        bodyPart = enums.BodyPartPlot.Trunk
    elif bodyPart == "UpperLimb":
        bodyPart = enums.BodyPartPlot.UpperLimb
    else:
        raise Exception("[pyCGM2] - bodyPart argument not recognized ( must be LowerLimb, Trunk or UpperLimb) ")


    kv = comparisonPlotViewers.KineticsPlotComparisonViewer(analyses,context,legends,bodyPart=bodyPart,pointLabelSuffix_lst=pointSuffixes)

    kv.setAutomaticYlimits(autoYlim)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-Kinetic comparison")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareReaction(DATA_PATH:str,analyses:List[Analysis],legends:List[str],normativeDataset:NormativeData,
    plotType="Descriptive",eventType:str="Gait",pointSuffixes=None,show:bool=True,title:Optional[str]=None,
    OUT_PATH:Optional[str]=None,outputName:Optional[str]=None,exportPng:bool=False,exportPdf:bool=False,autoYlim:bool=False):

    """
    Plots ground reaction from different analysis instances for comparison.

    This function visualizes and compares ground reaction data from multiple analysis instances. It supports 
    descriptive and consistency plot types and can compare data across different contexts .

    Args:
        DATA_PATH (str): Path to the data directory.
        analyses (List[Analysis]): List of Analysis instances to compare.
        legends (List[str]): Labels representing each analysis instance.
        normativeDataset (NormativeData): Normative data instance for comparison.
        plotType (str): Type of plot ('Descriptive' or 'Consistency'). Defaults to 'Descriptive'.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        pointSuffixes (Optional[List[str]]): Suffixes previously added to model outputs. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = compareReaction("/data/path", [analysis1, analysis2], ["pre", "post"],  normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "pyCGM2-Comparison"+"-"

    if exportPdf or exportPng:
        filenameOut =  outputName+"- Reaction Comparison "


    i=1
    for analysis in analyses:
        if analysis.kineticStats.data == {}:
            raise Exception("[pyCGM2]: Reaction comparison aborted. Analysis [%i] has no kinetic data"%(i))
        i+=1


    kv = comparisonPlotViewers.GroundReactionForceComparisonViewer(analyses,legends,pointLabelSuffix_lst=pointSuffixes)

    kv.setAutomaticYlimits(autoYlim)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if eventType =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-Reaction comparison")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareEmgEnvelops(DATA_PATH:str,analyses:List[Analysis],legends:List[str],
        normalized:bool=False,plotType="Descriptive",show:bool=True,title:Optional[str]=None,
        eventType:str="Gait",
        OUT_PATH:Optional[str]=None,outputName:Optional[str]=None,exportPng:bool=False,exportPdf:bool=False,autoYlim:bool=False,**kwargs):
    """
    Plots EMG envelopes from different analysis instances for comparison.

    This function visualizes and compares EMG data from multiple analysis instances. It supports 
    descriptive and consistency plot types and allows the comparison of EMG envelopes, 
    with options for normalization and event type specification.

    Args:
        DATA_PATH (str): Path to the data directory.
        analyses (List[Analysis]): List of Analysis instances to compare.
        legends (List[str]): Labels representing each analysis instance.
        normalized (bool): If True, displays normalized EMG amplitude. Defaults to False.
        plotType (str): Type of plot ('Descriptive' or 'Consistency'). Defaults to 'Descriptive'.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Keyword Args:
        forceEmgManager (Optional[pyCGM2.EMG.EmgManager]): Use a specific EmgManager instance.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = compareEmgEnvelops("/data/path", [analysis1, analysis2], ["pre", "post"])

    Note:
        The function is particularly useful for comparing EMG patterns across different conditions 
        or time points, providing insights into muscle activation consistency or variability.
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)
    emgChannels = emg.getChannels()


    if outputName is None:
        outputName = "pyCGM2-Comparison"

    if exportPdf or exportPng:
        filenameOut =  outputName+"- EMG Comparison"



    i=1
    for analysis in analyses:
        if analysis.emgStats.data == {}:
            raise Exception("[pyCGM2]: EMG comparison aborted. Analysis [%i] has no emg data"%(i))
        i+=1



    kv = emgPlotViewers.MultipleAnalysis_EnvEmgPlotPanelViewer(analyses,legends)

    kv.setAutomaticYlimits(autoYlim)

    kv.setEmgManager(emg)
    kv.selectEmgChannels(emgChannels)

    if normalized:
        kv.setNormalizedEmgFlag(True)

    if eventType=="Gait":
        if plotType == "Descriptive":
            kv.setConcretePlotFunction(plot.gaitDescriptivePlot )
        elif plotType == "Consistency":
            kv.setConcretePlotFunction(plot.gaitConsistencyPlot )
    else:
        if plotType == "Descriptive":
            kv.setConcretePlotFunction(plot.descriptivePlot )
        elif plotType == "Consistency":
            kv.setConcretePlotFunction(plot.consistencyPlot )


    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-EMG envelop comparison")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareSelectedEmgEvelops(DATA_PATH:str,analyses:List[Analysis],legends:List[str], emgChannels:List[str],contexts:List[str],
    normalized:bool=False,plotType:str="Descriptive",eventType:str="Gait",show:bool=True,
    title:Optional[str]=None,
    OUT_PATH =None, outputName:Optional[str]=None,exportPng:bool=False,exportPdf:bool=False,autoYlim:bool=False):
    """
    Compares selected EMG envelopes from different analysis instances constructed from the same session.

    This function visualizes and compares selected EMG channels from multiple analysis instances, 
    providing insights into muscle activation patterns. It supports normalization and different plot types.

    Args:
        DATA_PATH (str): Path to the data directory.
        analyses (List[Analysis]): List of Analysis instances to compare.
        legends (List[str]): Labels representing each analysis instance.
        emgChannels (List[str]): Names of EMG channels to compare.
        contexts (List[str]): Event contexts corresponding to the EMG channels.
        normalized (bool): If True, displays normalized EMG amplitude. Defaults to False.
        plotType (str): Type of plot ('Descriptive' or 'Consistency'). Defaults to 'Descriptive'.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = compareSelectedEmgEvelops("/data/path", [analysis1, analysis2], ["pre", "post"], ["EMG1", "EMG2"], ["Left", "Right"])
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "pyCGM2-Comparison"

    if exportPdf or exportPng:
        filenameOut =  outputName+"- specific EMG Comparison"

    fig = plt.figure()
    ax = plt.gca()

    colormap_i_left=[plt.cm.Reds(k) for k in np.linspace(0.2, 1, len(analyses))]
    colormap_i_right=[plt.cm.Blues(k) for k in np.linspace(0.2, 1, len(analyses))]

    i=0
    for analysis in analyses:
        label = emgChannels[i] + "_Rectify_Env" if not normalized else emgChannels[i] + "_Rectify_Env_Norm"
        title = "EMG Envelop Comparison" if not normalized else "Normalized EMG Envelop Comparison"

        if contexts[i] == "Left":
            color=colormap_i_left[i]
        elif contexts[i] == "Right":
            color=colormap_i_right[i]

        if plotType == "Descriptive":
            if eventType =="Gait":
                plot.gaitDescriptivePlot(ax,analysis.emgStats,
                                        label,contexts[i],0,
                                        color=color,
                                        title=title, xlabel="Gait Cycle", ylabel="emg",ylim=None,
                                        customLimits=None,legendLabel=legends[i])
            else:
                plot.descriptivePlot(ax,analysis.emgStats,
                                        label,contexts[i],0,
                                        color=None,
                                        title=title, xlabel="Gait Cycle", ylabel="emg",ylim=None,
                                        customLimits=None,legendLabel=legends[i])
        elif plotType == "Consistency":
            if eventType =="Gait":
                plot.gaitConsistencyPlot(ax,analysis.emgStats,
                                        label,contexts[i],0,
                                        color=color,
                                        title=title, xlabel="Gait Cycle", ylabel="emg",ylim=None,
                                        customLimits=None,legendLabel=legends[i])
            else:
                plot.consistencyPlot(ax,analysis.emgStats,
                                        label,contexts[i],0,
                                        color=None,
                                        title=title, xlabel="Gait Cycle", ylabel="emg",ylim=None,
                                        customLimits=None,legendLabel=legends[i])
        else:
            raise Exception ("[pyCGM2]: plot type does not recongnized")

        i+=1

        ax.legend(fontsize=6)
    if show: plt.show()

    if exportPdf:
        fig.savefig(OUT_PATH+filenameOut+".pdf")

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plot_DescriptiveMuscleLength(DATA_PATH:str,analysis:Analysis,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        normalizedSuffix=None,
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False,
        analyticalData=None,muscles=None):
    """
    Displays average and standard deviation of time-normalized muscle length outputs.

    This function visualizes the muscle length data from an analysis instance, compared against normative datasets. 
    It supports options for normalization, event type specification, and selective muscle analysis.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing muscle length data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        normalizedSuffix (Optional[str]): Suffix for normalized data. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.
        analyticalData (Optional[any]): Additional data for horizontal line plotting. Defaults to None.
        muscles (Optional[List[str]]): Specific muscles to analyze. Defaults to None.

    Returns:
        Tuple[List[matplotlib.figure.Figure], List[str]]: A list of Matplotlib figure objects and a list of output filenames.

    Examples:
        >>> figs, filenames = plot_DescriptiveMuscleLength("/data/path", analysisInstance, normativeDataset)
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    # detect muscle name in the analysis
    if muscles is None:
        detectedMuscles=[] 
        for name,side in  analysis.muscleGeometryStats.data:
            muscle = name[:name.find("[")-2]
            if muscle not in detectedMuscles: 
                detectedMuscles.append(muscle)
    else:
        detectedMuscles = muscles
    

    opensimMuscles_grouped = [detectedMuscles[i:i+16] for i in range(0, len(detectedMuscles), 16)]
   
    pageNumber = len(opensimMuscles_grouped)

    figs=[]
    outfilenames=[]

    exportFlag = True if exportPdf or exportPng else False

    page = 0
    for i in range(0,pageNumber):

        if outputName is None:
            outputName = "pyCGM2-analysis" 
  
        filenameOut =  outputName+"- descriptive muscleLength  ["+ str(page)+"]"

        # viewer
        kv =musclePlotViewers.MuscleNormalizedPlotPanelViewer(analysis)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        kv.setMuscles(opensimMuscles_grouped[page])
        kv.setMuscleOutputType("MuscleLength")
        if normalizedSuffix is not None: kv.setNormalizationSuffix(normalizedSuffix) 
        if normativeDataset is not None: kv.setNormativeDataset(normativeDataset)

        kv.setAutomaticYlimits(autoYlim)



        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        if title is not None: pf.setTitle(title+"-descriptive MuscleLength ["+ str(page)+"]")
        if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")


        fig = pf.plot()

        if analyticalData is not None: pf.setHorizontalLines(analyticalData)
        if exportPng:fig.savefig(OUT_PATH+filenameOut+".png")
        
        outfilenames.append(filenameOut+".png")
        figs.append(fig)

        page+=1
    
    if show: plt.show()

    return figs,outfilenames


def plot_ConsistencyMuscleLength(DATA_PATH:str,analysis:Analysis,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        normalizedSuffix=None,
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False,
        analyticalData=None,muscles=None):
    """
    Displays all cycle of time-normalized muscle length outputs.

    This function visualizes the muscle length data from an analysis instance, compared against normative datasets. 
    It supports options for normalization, event type specification, and selective muscle analysis.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing muscle length data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        normalizedSuffix (Optional[str]): Suffix for normalized data. Defaults to None.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.
        analyticalData (Optional[any]): Additional data for horizontal line plotting. Defaults to None.
        muscles (Optional[List[str]]): Specific muscles to analyze. Defaults to None.

    Returns:
        Tuple[List[matplotlib.figure.Figure], List[str]]: A list of Matplotlib figure objects and a list of output filenames.

    Examples:
        >>> figs, filenames = plot_ConsistencyMuscleLength("/data/path", analysisInstance, normativeDataset)
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    # detect muscle name in the analysis
    if muscles is None:
        detectedMuscles=[] 
        for name,side in  analysis.muscleGeometryStats.data:
            muscle = name[:name.find("[")-2]
            if muscle not in detectedMuscles: 
                detectedMuscles.append(muscle)
    else:
        detectedMuscles = muscles
    

    opensimMuscles_grouped = [detectedMuscles[i:i+16] for i in range(0, len(detectedMuscles), 16)]
   
    pageNumber = len(opensimMuscles_grouped)

    figs=[]
    outfilenames=[]

    exportFlag = True if exportPdf or exportPng else False

    page = 0
    for i in range(0,pageNumber):

        if outputName is None:
            outputName = "pyCGM2-analysis" 
  
        filenameOut =  outputName+"- consistency muscleLength  ["+ str(page)+"]"

        # viewer
        kv =musclePlotViewers.MuscleNormalizedPlotPanelViewer(analysis)
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
        kv.setMuscles(opensimMuscles_grouped[page])
        kv.setMuscleOutputType("MuscleLength")
        if normalizedSuffix is not None: kv.setNormalizationSuffix(normalizedSuffix) 
        if normativeDataset is not None: kv.setNormativeDataset(normativeDataset)

        kv.setAutomaticYlimits(autoYlim)



        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        if title is not None: pf.setTitle(title+"-consistency MuscleLength ["+ str(page)+"]")
        if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")


        fig = pf.plot()

        if analyticalData is not None: pf.setHorizontalLines(analyticalData)
        if exportPng:fig.savefig(OUT_PATH+filenameOut+".png")
        
        outfilenames.append(filenameOut+".png")
        figs.append(fig)

        page+=1
    
    if show: plt.show()

    return figs,outfilenames

def plotPFKE(DATA_PATH:str,analysisInstance:Analysis,normativeDataset:NormativeData,
    OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False):
    """
    Plots the PlantarFlexor-KneeExtensor (PFKE) index  based on an analysis instance.

    This function visualizes the PFKE, a classification based on normative datasets.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysisInstance (Analysis): An Analysis instance containing kinematic data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plotPFKE("/data/path", analysisInstance, normativeDataset)
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    procedure = classificationProcedures.PFKEprocedure(normativeDataset)
    filt = classificationFilters.ClassificationFilter(analysisInstance, procedure)
    sagClass = filt.run()
    classFig = procedure.plot(analysisInstance)

    if outputName is None:
        outputName = "pyCGM2-analysis"

    filenameOut =  outputName+"- pfke"

    if exportPng:plt.savefig(OUT_PATH+filenameOut+".png")
    if exportPdf:plt.savefig(OUT_PATH+filenameOut+".pdf")
    if show: plt.show()

    if exportPng:
        return classFig, filenameOut+".png"
    else:
        return classFig 

def plot_DescriptiveGRF(DATA_PATH:str,analysis:Analysis,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False):
    """display average and standard deviation of time-normalized ground reaction force.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (str)[Optional,None]:suffix previously added to your model outputs.
        eventType (str): [Optional, "Gait"]. event type. By default cycle is defined from foot strike.  `Gait` searched for the foot off events.
        OUT_PATH (str)[Optional,None]: path to your ouput folder
        exportPdf (bool)[Optional,False]: export as pdf
        outputName (str)[Optional,None]: name of the output filename.
        show (bool)[Optional,True]: show matplotlib figure.
        title (str)[Optional,None]: modify the plot panel title.
        exportPng (bool)[Optional,False]: export as png.
        autoYlim(bool)[Optional,False]: ignore predefined Y-axis boundaries

    Examples:

    .. code-block:: python

        plot_DescriptiveKinematic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    outputName = "pyCGM2-analysis ground reaction force"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-descriptive GRF "


    # filter 1 - descriptive kinematic panel
    #-------------------------------------------
    # viewer

    kv = groundReactionPlotViewers.NormalizedGroundReactionForcePlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-descriptive Ground reaction force")
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig
    

def plot_ConsistencyGRF(DATA_PATH:str,analysis:Analysis,normativeDataset:NormativeData,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False):
    """display all-cycle time-normalized ground reaction force.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (str)[Optional,None]:suffix previously added to your model outputs.
        eventType (str): [Optional, "Gait"]. event type. By default cycle is defined from foot strike.  `Gait` searched for the foot off events.
        OUT_PATH (str)[Optional,None]: path to your ouput folder
        exportPdf (bool)[Optional,False]: export as pdf
        outputName (str)[Optional,None]: name of the output filename.
        show (bool)[Optional,True]: show matplotlib figure.
        title (str)[Optional,None]: modify the plot panel title.
        exportPng (bool)[Optional,False]: export as png.
        autoYlim(bool)[Optional,False]: ignore predefined Y-axis boundaries

    Examples:

    .. code-block:: python

        plot_DescriptiveKinematic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    outputName = "pyCGM2-analysis ground reaction force"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-consistency GRF "


    # filter 1 - consistency kinematic panel
    #-------------------------------------------
    # viewer

    kv = groundReactionPlotViewers.NormalizedGroundReactionForcePlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-consistency Ground reaction force")
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig




 # # procedure - filter
        

def plot_DescriptiveGrfIntegration(DATA_PATH:str,analysis,normativeDataset,bodymass,
        pointLabelSuffix:Optional[str]=None,eventType:str="Gait",
        OUT_PATH:Optional[str]=None,exportPdf:bool=False,outputName:Optional[str]=None,show:bool=True,title:Optional[str]=None,exportPng:bool=False,
        autoYlim:bool=False):
    """
    Displays average and standard deviation of time-normalized ground reaction force integration.

    This function visualizes integrated GRF data from an analysis instance, offering a comprehensive 
    overview of force distribution throughout gait cycles. It compares against normative datasets, 
    considering the body mass of the subject.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing GRF data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        bodymass (float): The body mass of the subject.
        pointLabelSuffix (Optional[str]): Suffix previously added to model outputs. Defaults to None.
        eventType (str): Event type to consider (e.g., 'Gait'). Defaults to 'Gait'.
        OUT_PATH (Optional[str]): Path for saving exported files. Defaults to None.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        outputName (Optional[str]): Name of the output file. Defaults to None.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.

    Examples:
        >>> fig = plot_DescriptiveGrfIntegration("/data/path", analysisInstance, normativeDataset, 75.0)
    """
    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if outputName is None:
        outputName = "pyCGM2-analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-descriptive Force Plate Integration "


    # filter 1 - descriptive kinematic panel
    #-------------------------------------------
    # viewer


    kv = groundReactionPlotViewers.NormalizedGaitGrfIntegrationPlotViewer(analysis,bodymass = bodymass,pointLabelSuffix=None)
                                     
    kv.setAutomaticYlimits(autoYlim)

    if eventType == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)


    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-descriptive FP Integration")
    if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plotSaggitalGagePanel(DATA_PATH:str,
                          analysis:Analysis,
                          normativeDataset:NormativeData,
                          emgType:str="Envelope",
                          exportPdf:bool=False,
                          OUT_PATH:Optional[str]= None,
                          outputName=None,
                          show:bool=True,
                          title=None,exportPng=False,autoYlim:bool=False,**kwargs):
    """
    Creates and displays a saggital gait analysis plot as described in 'The Identification and Treatment of Gait Problems in Cerebral Palsy' by Gage et al.

    This function generates a plot that combines kinematic, kinetic, and electromyographic data to provide a comprehensive view of sagittal gait analysis.

    Args:
        DATA_PATH (str): Path to the data directory.
        analysis (Analysis): An Analysis instance containing the gait analysis data.
        normativeDataset (NormativeData): A NormativeData instance for comparison.
        emgType (str): type of emg signal to plot. Defaults to `Envelope`, choice: `Raw` or `Rectify`.
        exportPdf (bool): If True, exports the plot as a PDF. Defaults to False.
        OUT_PATH (Optional[str]): Path for saving exported files. If None, uses DATA_PATH. Defaults to None.
        outputName (Optional[str]): Name of the output file. Defaults to 'PyCGM2-Analysis'.
        show (bool): If True, shows the plot using Matplotlib. Defaults to True.
        title (Optional[str]): Title for the plot panel. Defaults to None.
        exportPng (bool): If True, exports the plot as a PNG. Defaults to False.
        autoYlim (bool): If True, sets Y-axis limits automatically. Defaults to False.
        **kwargs: Additional keyword arguments, including 'forceEmgManager' for specifying an EMG Manager.

    Returns:
        Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, str]]: The Matplotlib figure object. 
        If exporting as PNG, returns a tuple of the figure object and the filename.
    """

    if OUT_PATH is None:
        OUT_PATH = DATA_PATH


    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)
    
    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-SaggitalGageViewer"

    # viewer
    kv =customPlotViewers.SaggitalGagePlotViewer(analysis,emg,emgType=emgType)
    kv.setNormativeDataset(normativeDataset)
    kv.setAutomaticYlimits(autoYlim)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle( title+"-Gage Sagital Viewer")

    if exportPdf :pf.setExport(OUT_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(OUT_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig



# def plot_GaitMeanGrfIntegration(DATA_PATH,analysis,normativeDataset,
#         bodymass,
#         pointLabelSuffix=None,eventType="Gait",
#         OUT_PATH=None,exportPdf=False,outputName=None,show=True,title=None,exportPng=False,
#         autoYlim=False):
#     """display average and standard deviation of time-normalized ground reaction force.

#     Args:
#         DATA_PATH (str): path to your data
#         analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
#         bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
#         normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
#         pointLabelSuffix (str)[Optional,None]:suffix previously added to your model outputs.
#         eventType (str): [Optional, "Gait"]. event type. By default cycle is defined from foot strike.  `Gait` searched for the foot off events.
#         OUT_PATH (str)[Optional,None]: path to your ouput folder
#         exportPdf (bool)[Optional,False]: export as pdf
#         outputName (str)[Optional,None]: name of the output filename.
#         show (bool)[Optional,True]: show matplotlib figure.
#         title (str)[Optional,None]: modify the plot panel title.
#         exportPng (bool)[Optional,False]: export as png.
#         autoYlim(bool)[Optional,False]: ignore predefined Y-axis boundaries

#     Examples:

#     .. code-block:: python

#         plot_DescriptiveKinematic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

#     """
#     if OUT_PATH is None:
#         OUT_PATH = DATA_PATH

#     outputName = "pyCGM2-analysis ground reaction force integration"

#     if exportPdf or exportPng:
#         filenameOut =  outputName+"-descriptive GRF "


#     # filter 1 - descriptive kinematic panel
#     #-------------------------------------------
#     # viewer

#     kv = groundReactionPlotViewers.NormalizedGaitMeanGrfIntegrationPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix,bodymass=bodymass)
#     kv.setAutomaticYlimits(autoYlim)


#     if normativeDataset is not None:
#         kv.setNormativeDataset(normativeDataset)

#     # filter
#     pf = plotFilters.PlottingFilter()
#     pf.setViewer(kv)
#     if title is not None: pf.setTitle(title+"-descriptive Ground reaction force Integration")
#     if exportPdf: pf.setExport(OUT_PATH,filenameOut,"pdf")
#     fig = pf.plot()
#     if show: plt.show()

#     if exportPng:
#         fig.savefig(OUT_PATH+filenameOut+".png")
#         return fig,filenameOut+".png"
#     else:
#         return fig