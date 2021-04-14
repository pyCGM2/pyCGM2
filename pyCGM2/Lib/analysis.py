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

def processEMG(DATA_PATH, gaitTrials, emgChannels, highPassFrequencies=[20,200],envelopFrequency=6.0, fileSuffix=None,outDataPath=None):

    """
    processEMG_fromC3dFiles : filters emg channels from a list of c3d files

    :param DATA_PATH [String]: path to your folder
    :param gaitTrials [string List]:c3d files with emg signals
    :param emgChannels [string list]: label of your emg channels

    **optional**

    :param highPassFrequencies [list of float]: boundaries of the bandpass filter
    :param envelopFrequency [float]: cut-off frequency for creating an emg envelop
    :param fileSuffix [string]: suffix added to your ouput c3d files

    """
    if fileSuffix is None: fileSuffix=""

    for gaitTrial in gaitTrials:
        acq = btkTools.smartReader(DATA_PATH +gaitTrial)

        flag = False
        for channel in emgChannels:
            if not btkTools.isAnalogExist(acq,channel):
                LOGGER.logger.error( "channel [%s] not detected in the c3d [%s]"%(channel,gaitTrial))
                flag = True
        if flag:
            raise Exception ("[pyCGM2] One label has not been detected as analog. see above")

        bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
        bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
        envf.setCutoffFrequency(envelopFrequency)
        envf.run()

        outFilename = gaitTrial if fileSuffix=="" else gaitTrial[0:gaitTrial.rfind(".")]+"_"+fileSuffix+".c3d"

        if outDataPath is None:
            btkTools.smartWriter(acq,DATA_PATH+outFilename)
        else:
            btkTools.smartWriter(acq,outDataPath+outFilename)

def processEMG_fromBtkAcq(acq, emgChannels, highPassFrequencies=[20,200],envelopFrequency=6.0):
    """
    processEMG_fromBtkAcq : filt emg from a btk acq

    :param acq [btk::Acquisition]: btk acquisition
    :param emgChannels [string list]: label of your emg channels

    **optional**

    :param highPassFrequencies [list of float]: boundaries of the bandpass filter
    :param envelopFrequency [float]: cut-off frequency for creating an emg envelop

    """


    bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
    bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
    bf.run()

    envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
    envf.setCutoffFrequency(envelopFrequency)
    envf.run()

    return acq

def normalizedEMG(analysis, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=None, mvcSettings=None):
    """
    normalizedEMG : perform normalization of emg in amplitude

    :param analysis [pyCGM2.Processing.analysis.Analysis]: pyCGM2 analysis instance
    :param emgChannels [string list]: label of your emg channels
    :param contexts [string list]: contexts associated with your emg channel

    **optional**

    :param method [str]: method of amplitude normalisation (choice MeanMax[default], MaxMax, MedianMax)
    :param fromOtherAnalysis [pyCGM2.Processing.analysis.Analysis]: amplitude normalisation from another analysis instance

    """


    i=0
    for label in emgChannels:
        envnf = emgFilters.EmgNormalisationProcessingFilter(analysis,label,contexts[i])

        if fromOtherAnalysis is not None and mvcSettings is None:
            LOGGER.logger.info("[pyCGM2] - %s normalized from another Analysis"%(label))
            envnf.setThresholdFromOtherAnalysis(fromOtherAnalysis)

        if mvcSettings is not None:
            if label in mvcSettings.keys():
                LOGGER.logger.info("[pyCGM2] - %s normalized from MVC"%(label))
                envnf.setThresholdFromOtherAnalysis(mvcSettings[label])
            else:
                if fromOtherAnalysis is not None:
                    LOGGER.logger.info("[pyCGM2] - %s normalized from an external Analysis"%(label))
                    envnf.setThresholdFromOtherAnalysis(fromOtherAnalysis)
                else:
                    LOGGER.logger.info("[pyCGM2] - %s normalized from current analysis"%(label))

        if method != "MeanMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        elif method != "MaxMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MaxMax)
        elif method != "MedianMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MedianMax)

        envnf.run()
        i+=1
        del envnf

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
