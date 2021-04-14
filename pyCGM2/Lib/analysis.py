# -*- coding: utf-8 -*-
#import ipdb
import pyCGM2; LOGGER = pyCGM2.LOGGER
import pyCGM2
from pyCGM2.Processing import c3dManager, cycle, analysis
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2 import enums
from pyCGM2.Processing import exporter
from pyCGM2.Processing import jointPatterns


def makeAnalysis(DATA_PATH,
                    filenames,
                    type="Gait",
                    kinematicLabelsDict=cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT,
                    kineticLabelsDict=cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT,
                    emgChannels = ["Voltage.EMG1","Voltage.EMG2","Voltage.EMG3","Voltage.EMG4","Voltage.EMG5",
                                   "Voltage.EMG6","Voltage.EMG7","Voltage.EMG8","Voltage.EMG9","Voltage.EMG10",
                                   "Voltage.EMG11","Voltage.EMG12","Voltage.EMG13","Voltage.EMG14","Voltage.EMG15",
                                   "Voltage.EMG16"],
                    pointLabelSuffix=None,
                    btkAcqs=None,
                    subjectInfo=None, experimentalInfo=None,modelInfo=None,
                    ):

    """
    makeAnalysis : create the pyCGM2.Processing.analysis.Analysis instance

    :param DATA_PATH [str]: path to your data
    :param modelledFilenames [string list]: c3d files with model outputs


    **optional**





    """
    if filenames == []: filenames=None

    #---- c3d manager

    if btkAcqs is not None:
        c3dmanagerProcedure = c3dManager.UniqueBtkAcqSetProcedure(DATA_PATH,filenames,acqs=btkAcqs)

    else:
        c3dmanagerProcedure = c3dManager.DistinctC3dSetProcedure(DATA_PATH, filenames, filenames, filenames, filenames)

    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)

    cmf.enableKinematic(True) if kinematicLabelsDict is not None else cmf.enableKinematic(False)
    cmf.enableKinetic(True) if kineticLabelsDict is not None else cmf.enableKinetic(False)
    cmf.enableEmg(True) if emgChannels is not None else cmf.enableEmg(False)

    trialManager = cmf.generate()


    #----cycles
    if type == "Gait":
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                                   kinematicAcqs = trialManager.kinematic["Acqs"],
                                                   kineticAcqs = trialManager.kinetic["Acqs"],
                                                   emgAcqs=trialManager.emg["Acqs"])

    else:
        cycleBuilder = cycle.CyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                                   kinematicAcqs = trialManager.kinematic["Acqs"],
                                                   kineticAcqs = trialManager.kinetic["Acqs"],
                                                   emgAcqs=trialManager.emg["Acqs"])

    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()



    if emgChannels is not None:
        emgLabelList = [label+"_Rectify_Env" for label in emgChannels]
    else:
        emgLabelList = None

    if type == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix,
                                                      emgLabelList = emgLabelList)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix,
                                                      emgLabelList = emgLabelList)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis



def exportAnalysis(analysisInstance,DATA_PATH,name, mode="Advanced"):

    """
    exportAnalysis : export the pyCGM2.Processing.analysis.Analysis instance in a xls spreadsheet

    :param analysisInstance [pyCGM2.Processing.analysis.Analysis]: pyCGM2 analysis instance
    :param DATA_PATH [str]: path to your data
    :param name [string]: name of the output file

    **optional**

    :param mode [string]: structure of the output xls (choice: Advanced[Default] or Basic)

    .. note::

        the advanced xls organizes data by row ( one raw = on cycle)
        whereas the Basic mode exports each model output in a new sheet



    """

    exportFilter = exporter.XlsAnalysisExportFilter()
    exportFilter.setAnalysisInstance(analysisInstance)
    exportFilter.export(name, path=DATA_PATH,excelFormat = "xls",mode = mode)


def automaticCPdeviations(DATA_PATH,analysis,pointLabelSuffix=None,filterTrue=False, export=True, outputname ="Nieuwenhuys2017" ):
    """
    Detect gait deviation for CP according a Delphi Consensus (Nieuwenhuys2017 et al 2017)

    :param analysis [pyCGM2.Processing.analysis.Analysis]: pyCGM2 analysis instance

    """

    RULES_PATH = pyCGM2.PYCGM2_SETTINGS_FOLDER +"jointPatterns\\"
    rulesXls = RULES_PATH+"Nieuwenhuys2017.xlsx"
    jpp = jointPatterns.XlsJointPatternProcedure(rulesXls,pointSuffix=pointLabelSuffix)
    dpf = jointPatterns.JointPatternFilter(jpp, analysis)
    dataFrameValues = dpf.getValues()
    dataFramePatterns = dpf.getPatterns(filter = filterTrue)

    if export:
        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFrameValues])
        xlsExport.export((outputname+"_Data"), path=DATA_PATH)

        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFramePatterns])
        xlsExport.export((outputname+"_Patterns"), path=DATA_PATH)

    return dataFramePatterns
