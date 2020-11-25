# -*- coding: utf-8 -*-
#import ipdb
import logging
import pyCGM2
from pyCGM2.Processing import c3dManager, cycle, analysis
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2 import enums
from pyCGM2.Processing import exporter
from pyCGM2.Processing import jointPatterns


def makeCGMGaitAnalysis(DATA_PATH,modelledFilenames,
                        emgFilenames, emgChannels,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        pointLabelSuffix=None):
    """
    makeCGMGaitAnalysis : create the pyCGM2.Processing.analysis.Analysis instance with modelled c3d and  EMG c3d



    """

    if modelledFilenames == []: modelledFilenames=None
    if emgFilenames == []: emgFilenames=None

    c3dmanagerProcedure = c3dManager.DistinctC3dSetProcedure(DATA_PATH, modelledFilenames, modelledFilenames, modelledFilenames, emgFilenames)

    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableSpatioTemporal(True) if modelledFilenames is not None else cmf.enableSpatioTemporal(False)
    cmf.enableKinematic(True) if modelledFilenames is not None else cmf.enableKinematic(False)
    cmf.enableKinetic(True) if modelledFilenames is not None else cmf.enableKinetic(False)
    cmf.enableEmg(True) if emgFilenames is not None else cmf.enableEmg(False)
    trialManager = cmf.generate()


    #----cycles
    cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                           kinematicAcqs = trialManager.kinematic["Acqs"],
                                           kineticAcqs = trialManager.kinetic["Acqs"],
                                           emgAcqs=trialManager.emg["Acqs"])

    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    #----analysis
    kinematicLabelsDict = cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT
    kineticLabelsDict = cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT
    emgLabelList  = [label+"_Rectify_Env" for label in emgChannels]


    analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                          kinematicLabelsDict = kinematicLabelsDict,
                                          kineticLabelsDict = kineticLabelsDict,
                                          pointlabelSuffix = pointLabelSuffix,
                                          emgLabelList = emgLabelList)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis

def makeAnalysis(DATA_PATH,
                    modelledFilenames,
                    type="Gait",
                    subjectInfo=None, experimentalInfo=None,modelInfo=None,
                    pointLabelSuffix=None,
                    kinematicLabelsDict=None,
                    kineticLabelsDict=None,
                    disableKinetics=False,
                    btkAcqs=None):

    """
    makeAnalysis : create the pyCGM2.Processing.analysis.Analysis instance

    :param DATA_PATH [str]: path to your data
    :param modelledFilenames [string list]: c3d files with model outputs


    **optional**

    :param type [str]: process files with gait events if selected type is Gait
    :param subjectInfo [dict]:  dictionnary gathering info about the patient (name,dob...)
    :param experimentalInfo [dict]:  dictionnary gathering info about the  data session (orthosis, gait task,... )
    :param modelInfo [dict]:  dictionnary gathering info about the used model)
    :param pointLabelSuffix [string]: suffix previously added to your model outputs
    :param kinematicLabelsDict [dict]: dictionnary with two entries,Left and Right, pointing to kinematic model outputs you desire processes
    :param kineticLabelsDict [dict]: dictionnary with two entries,Left and Right, pointing to kinetic model outputs you desire processes
    :param disableKinetics [bool]: disable kinetics processing
    :param btkAcqs [bool]: force the use of a list of openma trials

    .. note::

        The dictionnaries (subjectInfo,experimentalInfo,modelInfo) is interesting
        if you want to find these information within the xls file




    """
    #---- c3d manager

    if btkAcqs is not None:
        c3dmanagerProcedure = c3dManager.UniqueBtkAcqSetProcedure(DATA_PATH,modelledFilenames,acqs=btkAcqs)

    else:
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFilenames)

    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableEmg(False)
    if disableKinetics: cmf.enableKinetic(False)

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



    #----analysis
    if kinematicLabelsDict is None:
        kinematicLabelsDict = cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT

    if kineticLabelsDict is None:
        kineticLabelsDict = cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT



    if type == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                      kinematicLabelsDict = kinematicLabelsDict,
                                                      kineticLabelsDict = kineticLabelsDict,
                                                      pointlabelSuffix = pointLabelSuffix)


    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo, model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis

    #files.saveAnalysis(analysis,DATA_PATH,"Save_and_openAnalysis")

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
                logging.error( "channel [%s] not detected in the c3d [%s]"%(channel,gaitTrial))
                flag = True
        if flag:
            raise Exception ("[pyCGM2] One label has not been detected as analog. see above")

        bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
        bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
        envf.setCutoffFrequency(envelopFrequency)
        envf.run()

        outFilename = gaitTrial if fileSuffix=="" else gaitTrial+"_"+fileSuffix

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

def makeEmgAnalysis(DATA_PATH,
                    processedEmgFiles,
                    emgChannels,
                    subjectInfo=None, experimentalInfo=None,
                    type="Gait",
                    btkAcqs = None
                    ):

    """
    makeEmgAnalysis : create the pyCGM2.Processing.analysis.Analysis instance with only EMG signals


    :param DATA_PATH [str]: path to your data
    :param processedEmgFiles [string list]: c3d files with emg processed outputs
    :param emgChannels [string list]: label of your emg channels

    **optional**

    :param subjectInfo [dict]:  dictionnary gathering info about the patient (name,dob...)
    :param experimentalInfo [dict]:  dictionnary gathering info about the  data session (orthosis, gait task,... )
    :param type [str]: process files with gait events if selected type is Gait
    :param btkAcqs [bool]: force the use of a list of openma trials
    """

    if btkAcqs is not None:
        c3dmanagerProcedure = c3dManager.UniqueBtkAcqSetProcedure(DATA_PATH,processedEmgFiles,acqs=btkAcqs)
    else:
        c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,processedEmgFiles)

    cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
    cmf.enableSpatioTemporal(False)
    cmf.enableKinematic(False)
    cmf.enableKinetic(False)
    cmf.enableEmg(True)
    trialManager = cmf.generate()

    #---- GAIT CYCLES FILTER
    #--------------------------------------------------------------------------

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

    emgLabelList  = [label+"_Rectify_Env" for label in emgChannels]

    if type == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = emgLabelList,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=None,
                                                      experimentalInfos=experimentalInfo)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                      kinematicLabelsDict = None,
                                                      kineticLabelsDict = None,
                                                      emgLabelList = emgLabelList,
                                                      subjectInfos=subjectInfo,
                                                      modelInfos=None,
                                                      experimentalInfos=experimentalInfo)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    analysisInstance = analysisFilter.analysis

    return analysisInstance

def normalizedEMG(analysis, emgChannels,contexts, method="MeanMax", fromOtherAnalysis=None):
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


        if fromOtherAnalysis is not None:
            envnf.setThresholdFromOtherAnalysis(fromOtherAnalysis)

        if method is not "MeanMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MeanMax)
        elif method is not "MaxMax":
            envnf.setMaxMethod(enums.EmgAmplitudeNormalization.MaxMax)
        elif method is not "MedianMax":
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
