# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Functions
#APIDOC["Draft"]=False
#--end--

from pyCGM2.Utils import files
from pyCGM2.Processing.JointPatterns import jointPatternFilters
from pyCGM2.Processing.JointPatterns import jointPatternProcedures
from pyCGM2.Processing import exporter
from pyCGM2.Model.CGM2 import cgm
# from pyCGM2.Processing import c3dManager
from pyCGM2.Processing.C3dManager import c3dManagerProcedures
from pyCGM2.Processing.C3dManager import c3dManagerFilters
from pyCGM2.Processing import cycle
from pyCGM2.Processing import analysis
import pyCGM2
LOGGER = pyCGM2.LOGGER


def makeAnalysis(DATA_PATH,
                 filenames,
                 type="Gait",
                 kinematicLabelsDict=cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT,
                 kineticLabelsDict=cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT,
                 emgChannels=pyCGM2.EMG_CHANNELS,
                 pointLabelSuffix=None,
                 subjectInfo=None, experimentalInfo=None, modelInfo=None,
                 **kwargs):
    """
    This function normalises data in time and returns an **Analysis Instance** ie a nested dictionary  containing
    spatiotemporal parameters, normalized kinematics, normalized kinetics and normalized EMG envelops from a list of c3d files.

    By default: the function calls :

    - kinematic and kinetic ouputs of the CGM
    - emg channels names Voltage.EMG1 to Voltage.EMG16

    You can also compute spatiotemporal parameters, normalized kinematics, normalized kinetics and normalized EMG envelops
    from different set of c3d files. For that, use the named arguments :

    - pstfilenames
    - kinematicfilenames
    - kineticfilenames
    - emgfilenames

    Args:
        DATA_PATH (str): folder path
        filenames (list): list of c3d files to normalize
        type (str)[Gait]: event type (choice : "Gait" or "unknown").
        kinematicLabelsDict (dict)[cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT]: dictionary containing kinematic data to normalize.
        kineticLabelsDict (dict)[cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT]: dictionary containing kinetic data to normalize.
        emgChannels (list)[channels of emg.settings]: list of emg channels
        pointLabelSuffix (str)[None]: suffix associated to point output
        subjectInfo (dict)[None]: dictionary with metadata information about the subject.
        experimentalInfo (dict)[None]: dictionary with metadata information about the expreiment.
        modelInfo (dict)[None]: dictionary with metadata information about the model.

    Keyword Arguments:
        btkAcqs (list of btk.Acquisition): btkAcq instances to process instead of calling c3d file.
        pstfilenames (list)[None]: list of c3d files used for computing spatiotemporal parameters
        kinematicfilenames (list)[None]: list of c3d files used to normalize kinematic data
        kineticfilenames (list)[None]: list of c3d files used to normalize kinetic data
        emgfilenames (list)[None]: list of c3d files used to normalize emg data

    Returns:
        analysisFilter.analysis (pyCGM2.Processing.analysis.Analysis): an analysis instance


    Examples:

    .. code-block:: python

        analysisInstance = analysis.makeAnalysis(DATA_PATH, [file1.c3d,"file2.c3d"])

    The code takes 2 c3d files, then time normalized kinematics, kinetics and emg.
    Kinematic and  kinetic labels  are the default CGM output labels.
    The Emg channels are defined in the emg.setting file


    .. code-block:: python

        analysisInstance2 = analysis.makeAnalysis(DATA_PATH, [file1.c3d,"file2.c3d"],
        ..........................................kinematicLabelsDict = {"Left": ["LHipAngles,LKneeAngles"], "Right": ["RHipAngles,RKneeAngles"]},
        ..........................................kineticLabelsDict = {"Left": ["LHipMoment,LKneePower"], "Right": ["RHipMoment,RKneeMoment"],
        ..........................................emgChannels = ["Voltage.EMG1","Voltage.EMG2","Voltage.EMG3"],
        ..........................................subjectInfo = {"Name":"Doe","Firstname":"John"},
        ..........................................experimentalInfo = {"Barefoot":"No"},
        ..........................................modelInfo = {"Model":"CGM1"})

    The code called specific model outputs and emg channels.
    In addition, the code also adds subject, experimental and model metadata.
    These information will be displayed in the exported spreadsheet.

    """

    if filenames == [] or filenames is None:
        filenames = None

    #---- c3d manager
    if "btkAcqs" in kwargs.keys() and kwargs["btkAcqs"] is not None:
        c3dmanagerProcedure = c3dManagerProcedures.UniqueBtkAcqSetProcedure(
            DATA_PATH, filenames, acqs=kwargs["btkAcqs"])

    else:
        pstfilenames = kwargs["pstfilenames"] if "pstfilenames" in kwargs.keys(
        ) else None
        kinematicfilenames = kwargs["kinematicfilenames"] if "kinematicfilenames" in kwargs.keys(
        ) else None
        kineticfilenames = kwargs["kineticfilenames"] if "kineticfilenames" in kwargs.keys(
        ) else None
        emgfilenames = kwargs["emgfilenames"] if "emgfilenames" in kwargs.keys(
        ) else None

        iPstFilenames = filenames if pstfilenames is None else pstfilenames
        iKinematicFilenames = filenames if kinematicfilenames is None else kinematicfilenames
        iKineticFilenames = filenames if kineticfilenames is None else kineticfilenames
        iEmgFilenames = filenames if emgfilenames is None else emgfilenames

        c3dmanagerProcedure = c3dManagerProcedures.DistinctC3dSetProcedure(
            DATA_PATH, iPstFilenames, iKinematicFilenames, iKineticFilenames, iEmgFilenames)

    cmf = c3dManagerFilters.C3dManagerFilter(c3dmanagerProcedure)

    cmf.enableKinematic(
        True) if kinematicLabelsDict is not None else cmf.enableKinematic(False)
    cmf.enableKinetic(
        True) if kineticLabelsDict is not None else cmf.enableKinetic(False)
    cmf.enableEmg(True) if emgChannels is not None else cmf.enableEmg(False)

    trialManager = cmf.generate()

    #----cycles
    if type == "Gait":
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                               kinematicAcqs=trialManager.kinematic["Acqs"],
                                               kineticAcqs=trialManager.kinetic["Acqs"],
                                               emgAcqs=trialManager.emg["Acqs"])

    else:
        cycleBuilder = cycle.CyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                           kinematicAcqs=trialManager.kinematic["Acqs"],
                                           kineticAcqs=trialManager.kinetic["Acqs"],
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
                                                       kinematicLabelsDict=kinematicLabelsDict,
                                                       kineticLabelsDict=kineticLabelsDict,
                                                       pointlabelSuffix=pointLabelSuffix,
                                                       emgLabelList=emgLabelList)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                   kinematicLabelsDict=kinematicLabelsDict,
                                                   kineticLabelsDict=kineticLabelsDict,
                                                   pointlabelSuffix=pointLabelSuffix,
                                                   emgLabelList=emgLabelList)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo,
                           model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis


def exportAnalysis(analysisInstance, DATA_PATH, name,
                   mode="Advanced"):
    """export an Analysis instance as excel spreadsheet.

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): Analysis instance.
        DATA_PATH (str):folder path
        name (str): name of your excel file.
        mode (str)[Advanced]: spreadsheet mode . ("Advanced or Basic")

    Example:

    .. code-block:: python

        exportAnalysis(AnalysisInstance, "c:\\DATA\\","johnDoe")


    """

    exportFilter = exporter.XlsAnalysisExportFilter()
    exportFilter.setAnalysisInstance(analysisInstance)
    exportFilter.export(name, path=DATA_PATH, excelFormat="xls", mode=mode)


def automaticCPdeviations(DATA_PATH, analysis, reference="Nieuwenhuys2017", pointLabelSuffix=None, filterTrue=False, export=True, outputname="Nieuwenhuys2017", language="-fr"):

    RULES_PATH = pyCGM2.PYCGM2_SETTINGS_FOLDER + "jointPatterns\\"
    rulesXls = RULES_PATH+reference+language+".xlsx"
    jpp = jointPatternProcedures.XlsJointPatternProcedure(
        rulesXls, pointSuffix=pointLabelSuffix)
    dpf = jointPatternFilters.JointPatternFilter(jpp, analysis)
    dataFrameValues = dpf.getValues()
    dataFramePatterns = dpf.getPatterns(filter=filterTrue)

    if export:
        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFrameValues])
        xlsExport.export((outputname+"_Data"), path=DATA_PATH)

        xlsExport = exporter.XlsExportDataFrameFilter()
        xlsExport.setDataFrames([dataFramePatterns])
        xlsExport.export((outputname+"_Patterns"), path=DATA_PATH)

    return dataFramePatterns
