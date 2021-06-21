# -*- coding: utf-8 -*-
"""
This module gathers convenient functions fro plotting Kinematic - Kinetic and EMG

"""
#import ipdb
import matplotlib.pyplot as plt
import numpy as np
from pyCGM2.Report import plot, plotFilters, plotViewers, normativeDatasets, emgPlotViewers, ComparisonPlotViewers
from pyCGM2.Processing import scores
from pyCGM2.Tools import btkTools
from pyCGM2 import enums

def plotTemporalKinematic(DATA_PATH, modelledFilenames,bodyPart, pointLabelSuffix=None, exportPdf=False,outputName=None,show=True,title=None,
                          btkAcq=None,exportPng=False):
    """plotTemporalKinematic : display temporal trace of the CGM kinematic outputs

    Args:
        DATA_PATH (str): path to your data
        modelledFilenames (str): name of your c3d including kinematic output
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        pointLabelSuffix (str): suffix previously added to your model outputs.
        exportPdf (bool): export as pdf (default: False).
        outputName (type): name of the output file .
        show (bool): show the matplotlib figure (default: True) .
        title (str): modify plot panel title
        btkAcq (btk.Acquisition): Description of parameter `btkAcq`.
        exportPng (bool):export as png (default: False).

    Returns:


    Examples:

        >>> plotTemporalKinematic("C:\\myDATA\\", "file1.c3d","LowerLimb")

    """


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
            filenameOut =  modelledFilenames+"-Temporal Kinematics ["+ bodyPart.name+"]"
        else:
            filenameOut =  outputName+"-Temporal Kinematics ["+ bodyPart.name+"]"

    if btkAcq is not None:
        acq = btkAcq
        btkTools.sortedEvents(acq)
    else:
        acq =btkTools.smartReader(DATA_PATH + modelledFilenames)

    kv = plotViewers.TemporalKinematicsPlotViewer(acq,pointLabelSuffix=pointLabelSuffix,bodyPart = bodyPart)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Temporal Kinematics ["+ bodyPart.name+"]")
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plotTemporalKinetic(DATA_PATH, modelledFilenames,bodyPart,pointLabelSuffix=None,exportPdf=False,outputName=None,show=True,title=None,
                        btkAcq=None,exportPng=False):

    """plotTemporalKinetic : display temporal trace of the CGM kinetic outputs

    Args:
        DATA_PATH (str): path to your data
        modelledFilenames (str): name of your c3d including kinematic output
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        pointLabelSuffix (str): suffix previously added to your model outputs.
        exportPdf (bool): export as pdf (default: False).
        outputName (type): name of the output file .
        show (bool): show the matplotlib figure (default: True) .
        title (str): modify plot panel title
        btkAcq (btk.Acquisition): btk acquisition use instead of `modelledFilenames`
        exportPng (bool):export as png (default: False).

    Returns:


    Examples:

        >>> plotTemporalKinetic("C:\\myDATA\\", "file1.c3d","LowerLimb")

    """

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

    if btkAcq is not None:
        acq = btkAcq
        btkTools.sortedEvents(acq)

    else:
        acq =btkTools.smartReader(DATA_PATH+modelledFilenames)

    kv = plotViewers.TemporalKineticsPlotViewer(acq,pointLabelSuffix=pointLabelSuffix,bodyPart = bodyPart)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Temporal Kinetics ["+ bodyPart.name+"]")
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plotTemporalEMG(DATA_PATH, processedEmgfile, emgSettings, rectify = True,
                    exportPdf=False,outputName=None,show=True,title=None,
                    btkAcq=None,ignoreNormalActivity= False,exportPng=False,OUT_PATH=None):
    """Display temporal traces of EMG signals

    Args:
        DATA_PATH (str): path to your data
        processedEmgfile (str): name of your c3d file with emg.
        emgSettings (str): content of the emg.setting file.
        rectify (bool): display rectify or raw signal (default: True).
        exportPdf (bool): export as pdf (default: False).
        outputName (str): name of the output file.
        show (bool): show the matplotlib figure (default: True) .
        title (str): modify the plot panel title.
        btkAcq (btk.Acquisition): btk acquisition use instead of `modelledFilenames`.
        ignoreNormalActivity (bool): disable display of normal activity in the background.
        exportPng (bool): export as png.
        OUT_PATH (str): specify an path different than the `DATA_PATH` to export plot

    Returns:


    Examples:

        >>> plotTemporalEMG("C:\\myDATA\\", "file1.c3d", emgSettings)

    """


    if OUT_PATH is None:
        OUT_PATH = DATA_PATH

    if btkAcq is not None:
        acq = btkAcq
    else:
        acq =btkTools.smartReader(DATA_PATH+processedEmgfile)

    emgChannels = list()
    for channel in emgSettings["CHANNELS"].keys():
        if emgSettings["CHANNELS"][channel]["Muscle"] is not None:
            emgChannels.append(channel)
    emgChannels_list=  [emgChannels[i:i+10] for i in range(0, len(emgChannels), 10)]


    pageNumber = len(emgChannels_list)

    figs=list()
    outfilenames=list()

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
        kv.setEmgSettings(emgSettings)
        kv.selectEmgChannels(emgChannels_list[i])
        kv.ignoreNormalActivty(ignoreNormalActivity)
        kv. setEmgRectify(rectify)

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

def plotDescriptiveEnvelopEMGpanel(DATA_PATH,analysis, emgChannels, muscles,contexts, normalActivityEmgs, normalized=False,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):
    """ display average and standard deviation of time-normalized EMG envelops.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        emgChannels (list): names of your emg channels ( ie analog labels ).
        muscles (list): names of the muscles associated to each channel.
        contexts (list): event context (Left or Right) used to display the emg envelop. It makes sense to use "Left" for event context if the emg was placed on the left RECFEM
        normalActivityEmgs (list): muscle used as reference for displaying normal activity in the background.
        normalized (bool): enable plot of emg normalized in amplitude (default:False).
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:

    Examples:

        >>> plotDescriptiveEnvelopEMGpanel("C:\\myDATA\\", analysisInstance, ["Voltage.EMG1","Voltage.EMG1"], ["RECFEM","VASLAT"], ["Left","Right"], ["RECFEM","VASLAT"])

    """


    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-DescriptiveEmgEnv"+"[No Normalized]-" if not normalized else outputName+"-DescriptiveEmgEnv"+"[Normalized]"

    # viewer
    combinedEMGcontext=[]
    for i in range(0,len(emgChannels)):
        combinedEMGcontext.append([emgChannels[i],contexts[i], muscles[i]])


    kv = emgPlotViewers.EnvEmgGaitPlotPanelViewer(analysis)
    kv.setEmgs(combinedEMGcontext)
    kv.setNormalActivationLabels(normalActivityEmgs)
    kv.setNormalizedEmgFlag(normalized)

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)

    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)

    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plotConsistencyEnvelopEMGpanel(DATA_PATH,analysis, emgChannels,muscles, contexts, normalActivityEmgs, normalized=False,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):

    """ display all-cycles of time-normalized EMG envelops.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        emgChannels (list): names of your emg channels ( ie analog labels ).
        muscles (list): names of the muscles associated to each channel.
        contexts (list): event context (Left or Right) used to display the emg envelop. It makes sense to use "Left" for event context if the emg was placed on the left RECFEM
        normalActivityEmgs (list): muscle used as reference for displaying normal activity in the background.
        normalized (bool): enable plot of emg normalized in amplitude (default:False).
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plotConsistencyEnvelopEMGpanel("C:\\myDATA\\", analysisInstance, ["Voltage.EMG1","Voltage.EMG1"], ["RECFEM","VASLAT"], ["Left","Right"], ["RECFEM","VASLAT"])

    """


    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-ConsistencyEmgEnv"+"[No Normalized]-" if not normalized else outputName+"-DescriptiveEmgEnv"+"[Normalized]"

    # viewer
    combinedEMGcontext=[]
    for i in range(0,len(emgChannels)):
        combinedEMGcontext.append([emgChannels[i],contexts[i], muscles[i]])


    kv = emgPlotViewers.EnvEmgGaitPlotPanelViewer(analysis)
    kv.setEmgs(combinedEMGcontext)
    kv.setNormalActivationLabels(normalActivityEmgs)
    kv.setNormalizedEmgFlag(normalized)

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)

    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle( title+"-ConsistencyEmgEnv"+"[No Normalized]-" if not normalized else title+"-DescriptiveEmgEnv"+"[Normalized]")
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plot_spatioTemporal(DATA_PATH,analysis,exportPdf=False,outputName=None,show=True,title=None,exportPng=False):
    """display spatio-temporal parameters as horizontal histogram.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_spatioTemporal("C:\\myDATA\\", analysisInstance)

    """


    if outputName is None:  outputName = "pyCGM2-analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-SpatioTemporal parameters"

    stpv = plotViewers.SpatioTemporalPlotViewer(analysis)
    stpv.setNormativeDataset(normativeDatasets.NormalSTP())

    # filter
    stppf = plotFilters.PlottingFilter()
    stppf.setViewer(stpv)
    if title is not None: stppf.setTitle(title+"-SpatioTemporal parameters")
    if exportPdf: stppf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = stppf.plot()

    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_DescriptiveKinematic(DATA_PATH,analysis,bodyPart,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):
    """display average and standard deviation of time-normalized kinematic output.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (type):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_DescriptiveKinematic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """


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

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-descriptive Kinematics ["+ bodyPart.name+"]")
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig


def plot_ConsistencyKinematic(DATA_PATH,analysis,bodyPart,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):

    """display all cycles of time-normalized kinematic output.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (type):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_ConsistencyKinematic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """

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

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-consistency  Kinematics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_DescriptiveKinetic(DATA_PATH,analysis,bodyPart,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):
    """display average and standard deviation of time-normalized kinetic outputs.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (type):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_DescriptiveKinetic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """


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

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
    else:
        kv.setConcretePlotFunction(plot.descriptivePlot)



    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-descriptive  Kinetics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_ConsistencyKinetic(DATA_PATH,analysis,bodyPart, normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None,show=True,title=None,exportPng=False):
    """display all cycles of time-normalized kinetic outputs.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (type):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_ConsistencyKinetic("c:\\mydata\\",analysisInstance,"LowerLimb",normativeInstance)

    """

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

    if type == "Gait":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot)
    else:
        kv.setConcretePlotFunction(plot.consistencyPlot)

    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-consistency  Kinetics ["+ bodyPart.name+"]")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def plot_MAP(DATA_PATH,analysis,normativeDataset,exportPdf=False,outputName=None,pointLabelSuffix=None,show=True,title=None,exportPng=False):
    """display histogram of the Movement Analysis Profile.

    Args:
        DATA_PATH (str): path to your data
        analysis (pyCGM2.Processing.analysis.Analysis): analysis instance.
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        pointLabelSuffix (type):suffix previously added to your model outputs.
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> plot_MAP("c:\\mydata\\",analysisInstance,normativeInstance)

    """
    if outputName is None:
        outputName = "PyCGM2-Analysis"

    if exportPdf or exportPng:
        filenameOut =  outputName+"-Map"

    #compute
    gps =scores.CGM1_GPS(pointSuffix=pointLabelSuffix)
    scf = scores.ScoreFilter(gps,analysis, normativeDataset)
    scf.compute()

    #plot
    kv = plotViewers.GpsMapPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)

    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-Map")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareKinematic(DATA_PATH,analyses,legends,context,bodyPart,normativeDataset,plotType="Descriptive",type="Gait",pointSuffixes=None,show=True,title=None,outputName=None,exportPng=False,exportPdf=False):
    """plot kinematics from different analysis instances.

    Args:
        DATA_PATH (str): path to your data
        analysis (list): list of analysis instances.
        legends (list): short label representing each analysis instances
        context (str): event context
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        plotType (str): descriptive (ie average + sd) or consistency plots ( choice: Descriptive, Consistency)
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        pointSuffixes (list):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> compareKinematic("c:\\mydata\\",[analysisInstance1,analysisInstance2],["pre","post"],"Left","LowerLimb",normativeInstance)
    """


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

    kv = ComparisonPlotViewers.KinematicsPlotComparisonViewer(analyses,context,legends,bodyPart=bodyPart,pointLabelSuffix_lst=pointSuffixes)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if title is not None: pf.setTitle(title+"-Kinematic comparison")
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareKinetic(DATA_PATH,analyses,legends,context,bodyPart,normativeDataset,plotType="Descriptive",type="Gait",pointSuffixes=None,show=True,title=None,outputName=None,exportPng=False,exportPdf=False):

    """plot kinetics from different analysis instances.

    Args:
        DATA_PATH (str): path to your data
        analysis (list): list of analysis instances.
        legends (list): short label representing each analysis instances
        context (str): event context
        bodyPart (str): body part (choice : LowerLimb, Trunk, UpperLimb)
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): normative data instance.
        plotType (str): descriptive (ie average + sd) or consistency plots ( choice: Descriptive, Consistency)
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        pointSuffixes (list):suffix previously added to your model outputs.
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> compareKinetic("c:\\mydata\\",[analysisInstance1,analysisInstance2],["pre","post"],"Left","LowerLimb",normativeInstance)
    """
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


    kv = ComparisonPlotViewers.KineticsPlotComparisonViewer(analyses,context,legends,bodyPart=bodyPart,pointLabelSuffix_lst=pointSuffixes)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-Kinetic comparison")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareEmgEnvelops(DATA_PATH,analyses,legends, emgChannels, muscles, contexts, normalActivityEmgs, normalized=False,plotType="Descriptive",show=True,title=None,type="Gait",outputName=None,exportPng=False,exportPdf=False):
    """plot EMG envelops from different analysis instances.

    Args:
        DATA_PATH (str): path to your data
        analysis (list): list of analysis instances.
        legends (list): short label representing each analysis instances
        emgChannels (list): names of your emg channels ( ie analog labels ).
        muscles (list): names of the muscles associated to each channel.
        contexts (list): event context (Left or Right) used to display the emg envelop. It makes sense to use "Left" for event context if the emg was placed on the left RECFEM
        normalActivityEmgs (list): muscle used as reference for displaying normal activity in the background.
        normalized (bool): enable plot of emg normalized in amplitude (default:False).
        plotType (str): descriptive (ie average + sd) or consistency plots ( choice: Descriptive, Consistency)
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        >>> compareEmgEnvelops("c:\\mydata\\",[analysisInstance1,analysisInstance2],["pre","post"], ["Voltage.EMG1","Voltage.EMG1"], ["RECFEM","VASLAT"], ["Left","Right"], ["RECFEM","VASLAT"])
    """

    if outputName is None:
        outputName = "pyCGM2-Comparison"

    if exportPdf or exportPng:
        filenameOut =  outputName+"- EMG Comparison"



    i=1
    for analysis in analyses:
        if analysis.emgStats.data == {}:
            raise Exception("[pyCGM2]: EMG comparison aborted. Analysis [%i] has no emg data"%(i))
        i+=1


    combinedEMGcontext=[]
    for i in range(0,len(emgChannels)):
        combinedEMGcontext.append([emgChannels[i],contexts[i],muscles[i]])


    kv = emgPlotViewers.MultipleAnalysis_EnvEmgPlotPanelViewer(analyses,legends)

    kv.setEmgs(combinedEMGcontext)
    kv.setNormalActivationLabels(normalActivityEmgs)
    if normalized:
        kv.setNormalizedEmgFlag(True)

    if type=="Gait":
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
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    if title is not None: pf.setTitle(title+"-EMG envelop comparison")
    fig = pf.plot()
    if show: plt.show()

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig

def compareSelectedEmgEvelops(DATA_PATH,analyses,legends, emgChannels,contexts, normalized=False,plotType="Descriptive",type="Gait",show=True,title=None,outputName=None,exportPng=False,exportPdf=False):
    """compare selected EMG envelops from different analysis instances.

    Args:
        DATA_PATH (str): path to your data
        analysis (list): list of analysis instances.
        legends (list): short label representing each analysis instances
        emgChannels (list): names of your emg channels ( ie analog labels ).
        muscles (list): names of the muscles associated to each channel.
        contexts (list): event context (Left or Right) used to display the emg envelop. It makes sense to use "Left" for event context if the emg was placed on the left RECFEM
        normalized (bool): enable plot of emg normalized in amplitude (default:False).
        plotType (str): descriptive (ie average + sd) or consistency plots ( choice: Descriptive, Consistency)
        type (str): type of events (default: Gait). if different to Gait, use foot strike only to define cycles
        exportPdf (bool): export as pdf
        outputName (str): name of the output filename.
        show (bool): show matplotlib figure.
        title (str): modify the plot panel title.
        exportPng (bool): export as png.

    Returns:


    Examples:

        The following code plots the channel *Voltage.EMG1* time-normalized according *Left* events included in *analysisInstance1* with
        *Voltage.EMG2* time-normalized according *Left* events included in  *analysisInstance2*.

        >>> compareSelectedEmgEvelops("c:\\mydata\\",[analysisInstance1,analysisInstance2],["pre","post"], ["Voltage.EMG1","Voltage.EMG2"], ["Left","Left"])
    """



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
            if type =="Gait":
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
            if type =="Gait":
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
        fig.savefig(DATA_PATH+filenameOut+".pdf")

    if exportPng:
        fig.savefig(DATA_PATH+filenameOut+".png")
        return fig,filenameOut+".png"
    else:
        return fig
