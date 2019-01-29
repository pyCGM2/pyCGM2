# -*- coding: utf-8 -*-
#import ipdb
import matplotlib.pyplot as plt
import numpy as np
from pyCGM2.Report import plot, plotFilters, plotViewers, normativeDatasets, emgPlotViewers
from pyCGM2.Processing import scores
from pyCGM2.Tools import trialTools

def plotTemporalKinematic(DATA_PATH, modelledFilenames,pointLabelSuffix=None, exportPdf=False):

    if exportPdf:
        filenameOut =  str(modelledFilenames+"-Temporal Kinematics")

    trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames)

    kv = plotViewers.TemporalGaitKinematicsPlotViewer(trial,pointLabelSuffix=pointLabelSuffix)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    pf.plot()

    plt.show()


def plotTemporalKinetic(DATA_PATH, modelledFilenames,pointLabelSuffix=None,exportPdf=False):

    if exportPdf:
        filenameOut =  str(modelledFilenames+"-Temporal Kinetics")

    trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames)

    kv = plotViewers.TemporalGaitKineticsPlotViewer(trial,pointLabelSuffix=pointLabelSuffix)
    # # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    pf.plot()

    plt.show()

def plotTemporalEMG(DATA_PATH, processedEmgfile, emglabels, muscles, contexts, normalActivityEmgs, rectify = True,exportPdf=False):

    trial =trialTools.smartTrialReader(DATA_PATH,processedEmgfile)

    if len(emglabels)>10:
        pages = [[0,9], [10,15]]
    else:
        pages = [[0,9]]

    count = 0
    for page in pages:

        if exportPdf and len(page)>1:
            filenameOut =  str(processedEmgfile+"-TemporalEmgPlot"+"[rectify]-")+str(count) if rectify else str(processedEmgfile+"-TemporalEmgPlot"+"[raw]-")+str(count)
        else:
            filenameOut =  str(processedEmgfile+"-TemporalEmgPlot"+"[rectify]") if rectify else str(processedEmgfile+"-TemporalEmgPlot"+"[raw]")


        combinedEMGcontext=[]
        #for i in range(0,len(emglabels)): #len(emglabels
        for i in range(page[0],page[1]+1): #len(emglabels
            combinedEMGcontext.append([emglabels[i],contexts[i], muscles[i]])

        # # viewer
        kv = emgPlotViewers.TemporalEmgPlotViewer(trial)
        kv.setEmgs(combinedEMGcontext)
        kv.setNormalActivationLabels(normalActivityEmgs)
        kv. setEmgRectify(rectify)

        # # filter

        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
        pf.plot()

        plt.show()
        count+=1


def plotDescriptiveEnvelopEMGpanel(DATA_PATH,analysis, emglabels, muscles,contexts, normalActivityEmgs, normalized=False,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-DescriptiveEmgEnv"+"[No Normalized]-") if not normalized else str(outputName+"-DescriptiveEmgEnv"+"[Normalized]")

    # viewer
    combinedEMGcontext=[]
    for i in range(0,len(emglabels)):
        combinedEMGcontext.append([emglabels[i],contexts[i], muscles[i]])


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
    pf.plot()

    plt.show()


def plotConsistencyEnvelopEMGpanel(DATA_PATH,analysis, emglabels,muscles, contexts, normalActivityEmgs, normalized=False,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-ConsistencyEmgEnv"+"[No Normalized]-") if not normalized else str(outputName+"-DescriptiveEmgEnv"+"[Normalized]")

    # viewer
    combinedEMGcontext=[]
    for i in range(0,len(emglabels)):
        combinedEMGcontext.append([emglabels[i],contexts[i], muscles[i]])


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
    if exportPdf :pf.setExport(DATA_PATH,filenameOut,"pdf")
    pf.plot()

    plt.show()



def plot_spatioTemporal(DATA_PATH,analysis,exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-SpatioTemporal parameters")

    stpv = plotViewers.SpatioTemporalPlotViewer(analysis)
    stpv.setNormativeDataset(normativeDatasets.NormalSTP())

    # filter
    stppf = plotFilters.PlottingFilter()
    stppf.setViewer(stpv)
    if exportPdf: stppf.setExport(DATA_PATH,filenameOut,"pdf")
    stppf.plot()
    plt.show()

def plot_DescriptiveKinematic(DATA_PATH,analysis,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-descriptive  Kinematics")


    # filter 1 - descriptive kinematic panel
    #-------------------------------------------
    # viewer
    if analysis.modelInfo["Version"] in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.3","CGM2.4"]:
        kv = plotViewers.LowerLimbKinematicsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    # elif analysis.modelInfo["Version"] in ["CGM2.4"]:
    #     kv = plotViewers.LowerLimbMultiFootKinematicsPlotViewer(analysis,
    #                         pointLabelSuffix=pointLabelSuffix)
    else:
        raise Exception("[pyCGM2] Model version not known")

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
    pf.plot()
    plt.show()


def plot_ConsistencyKinematic(DATA_PATH,analysis,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-consistency Kinematics")

    if analysis.modelInfo["Version"] in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.3","CGM2.4"]:
        kv = plotViewers.LowerLimbKinematicsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)
    else:
        raise Exception("[pyCGM2] Model version not known")

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
    pf.plot()
    plt.show()


def plot_DescriptiveKinetic(DATA_PATH,analysis,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-descriptive Kinetics")

    if  analysis.modelInfo["Version"] in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.3","CGM2.4"]:
        kv = plotViewers.LowerLimbKineticsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)

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
    pf.plot()
    plt.show()


def plot_ConsistencyKinetic(DATA_PATH,analysis,normativeDataset,pointLabelSuffix=None,type="Gait",exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-consistency Kinetics")

    if  analysis.modelInfo["Version"] in["CGM1.0","CGM1.1","CGM2.1","CGM2.2","CGM2.3","CGM2.4"]:
        kv = plotViewers.LowerLimbKineticsPlotViewer(analysis,pointLabelSuffix=pointLabelSuffix)

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
    pf.plot()
    plt.show()

def plot_MAP(DATA_PATH,analysis,normativeDataset,exportPdf=False,outputName=None):

    if outputName is None:
        outputName = "Global Analysis"

    if exportPdf:
        filenameOut =  str(outputName+"-Map")

    #compute
    gps =scores.CGM1_GPS()
    scf = scores.ScoreFilter(gps,analysis, normativeDataset)
    scf.compute()

    #plot
    kv = plotViewers.GpsMapPlotViewer(analysis)

    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    if exportPdf: pf.setExport(DATA_PATH,filenameOut,"pdf")
    pf.plot()
    plt.show()


def compareKinematic(analyses,labels,context,normativeDataset,plotType="Descriptive",type="Gait"):


    kv = plotViewers.multipleAnalyses_LowerLimbKinematicsPlotViewer(analyses,context,labels)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    #pf.setExport(outputPath,str(pdfFilename+"-consisntency Kinematics"),"pdf")
    pf.plot()
    plt.show()


def compareKinetic(analyses,labels,context,normativeDataset,plotType="Descriptive",type="Gait"):


    kv = plotViewers.multipleAnalyses_LowerLimbKineticsPlotViewer(analyses,context,labels)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.descriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot ) if type =="Gait" else kv.setConcretePlotFunction(plot.consistencyPlot )


    if normativeDataset is not None:
        kv.setNormativeDataset(normativeDataset)

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    #pf.setExport(outputPath,str(pdfFilename+"-consisntency Kinematics"),"pdf")
    pf.plot()
    plt.show()


def compareEmgEvelops(analyses,legends, emglabels, contexts, normalActivityEmgs, normalized=False,plotType="Descriptive"):

    combinedEMGcontext=[]
    for i in range(0,len(emglabels)):
        combinedEMGcontext.append([emglabels[i],contexts[i]])


    kv = emgPlotViewers.MultipleAnalysis_EnvEmgGaitPlotPanelViewer(analyses,legends)

    kv.setEmgs(combinedEMGcontext)
    kv.setNormalActivationLabels(normalActivityEmgs)
    if normalized:
        kv.setNormalizedEmgFlag(True)

    if plotType == "Descriptive":
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot )
    elif plotType == "Consistency":
        kv.setConcretePlotFunction(plot.gaitConsistencyPlot )

    # filter
    pf = plotFilters.PlottingFilter()
    pf.setViewer(kv)
    #pf.setExport(outputPath,str(pdfFilename+"-consisntency Kinematics"),"pdf")
    pf.plot()
    plt.show()

def compareOneEmgEvelops(analyses,EMG_labels,contexts,legends, normalized=False,plotType="Descriptive",type="Gait"):

        fig = plt.figure()
        ax = plt.gca()

        colormap_i_left=[plt.cm.Reds(k) for k in np.linspace(0.2, 1, len(analyses))]
        colormap_i_right=[plt.cm.Blues(k) for k in np.linspace(0.2, 1, len(analyses))]

        i=0
        for analysis in analyses:
            label = EMG_labels[i] + "_Rectify_Env" if not normalized else EMG_labels[i] + "_Rectify_Env_Norm"
            title = "EMG Envelop Comparison" if not normalized else "Normalized EMG Envelop Comparison"

            if contexts[i] == "Left":
                color=colormap_i_left[i]
            elif contexts[i] == "Right":
                color=colormap_i_right[i]

            plot.gaitDescriptivePlot(ax,analysis.emgStats,
                                    label,contexts[i],0,
                                    color=color,
                                    title=title, xlabel="Gait Cycle", ylabel="emg",ylim=None,
                                    customLimits=None,legendLabel=legends[i])
            i+=1

            ax.legend(fontsize=6)
        plt.show()
