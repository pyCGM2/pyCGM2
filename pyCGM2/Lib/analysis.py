# -*- coding: utf-8 -*-

from pyCGM2.Utils import files
from pyCGM2.Processing.JointPatterns import jointPatternFilters
from pyCGM2.Processing.JointPatterns import jointPatternProcedures
from pyCGM2.Processing import exporter
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Processing.C3dManager import c3dManagerProcedures
from pyCGM2.Processing.C3dManager import c3dManagerFilters
from pyCGM2.Processing import cycle
from pyCGM2.Processing import analysis
from pyCGM2.Processing.analysis import Analysis

import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union

def makeAnalysis(DATA_PATH:str,
                 filenames:List,
                 eventType:str="Gait",
                 kinematicLabelsDict:Optional[Dict]=cgm.CGM.ANALYSIS_KINEMATIC_LABELS_DICT,
                 kineticLabelsDict:Optional[Dict]=cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT,
                 emgChannels:Optional[List]=pyCGM2.EMG_CHANNELS,
                 geometryMuscleLabelsDict:Optional[Dict]=None,
                 dynamicMuscleLabelsDict:Optional[Dict]=None,
                 pointLabelSuffix:Optional[str]=None,
                 subjectInfo:Optional[Dict]=None, 
                 experimentalInfo:Optional[Dict]=None, 
                 modelInfo:Optional[Dict]=None,
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
        DATA_PATH (str): Folder path containing c3d files.
        filenames (List[str]): List of c3d filenames for analysis.
        eventType (str, optional): Type of events, defaults to 'Gait'. Options include 'Gait' and 'unknown'.
        kinematicLabelsDict (Dict, optional): Dictionary specifying kinematic data to normalize.
        kineticLabelsDict (Dict, optional): Dictionary specifying kinetic data to normalize.
        emgChannels (List[str], optional): List of EMG channels. Defaults to channels defined in pyCGM2.EMG_CHANNELS.
        geometryMuscleLabelsDict (Optional[Dict], optional): Dictionary specifying muscle geometry labels.
        dynamicMuscleLabelsDict (Optional[Dict], optional): Dictionary specifying dynamic muscle labels.
        pointLabelSuffix (Optional[str], optional): Suffix associated with point outputs.
        subjectInfo (Optional[Dict], optional): Metadata information about the subject.
        experimentalInfo (Optional[Dict], optional): Metadata information about the experiment.
        modelInfo (Optional[Dict], optional): Metadata information about the model.

    Keyword Arguments:
        btkAcqs (List[btk.btkAcquisition]): Optional list of btkAcquisition instances to process.
        pstfilenames (List[str]): Optional list of c3d files for computing spatiotemporal parameters.
        kinematicfilenames (List[str]): Optional list of c3d files for normalizing kinematic data.
        kineticfilenames (List[str]): Optional list of c3d files for normalizing kinetic data.
        emgfilenames (List[str]): Optional list of c3d files for normalizing EMG data.


    Returns:
        analysis.Analysis: an analysis instance


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
        muscleGeometryfilenames = kwargs["muscleGeometryfilenames"] if "muscleGeometryfilenames" in kwargs.keys(
        ) else None
        muscleDynamicfilenames = kwargs["muscleDynamicfilenames"] if "muscleDynamicfilenames" in kwargs.keys(
        ) else None


        iPstFilenames = filenames if pstfilenames is None else pstfilenames
        iKinematicFilenames = filenames if kinematicfilenames is None else kinematicfilenames
        iKineticFilenames = filenames if kineticfilenames is None else kineticfilenames
        iEmgFilenames = filenames if emgfilenames is None else emgfilenames

        iMuscleGeometryFilenames = filenames if muscleGeometryfilenames is None else muscleGeometryfilenames
        iMuscleDynamicFilenames = filenames if muscleDynamicfilenames is None else muscleDynamicfilenames

        c3dmanagerProcedure = c3dManagerProcedures.DistinctC3dSetProcedure(
            DATA_PATH, iPstFilenames, iKinematicFilenames, iKineticFilenames, iEmgFilenames,
            iMuscleGeometryFilenames,iMuscleDynamicFilenames)

    cmf = c3dManagerFilters.C3dManagerFilter(c3dmanagerProcedure)

    cmf.enableKinematic(
        True) if kinematicLabelsDict is not None else cmf.enableKinematic(False)
    cmf.enableKinetic(
        True) if kineticLabelsDict is not None else cmf.enableKinetic(False)
    cmf.enableEmg(True) if emgChannels is not None else cmf.enableEmg(False)


    cmf.enableMuscleGeometry(True) if geometryMuscleLabelsDict is not None else cmf.enableMuscleGeometry(False)
    cmf.enableMuscleDynamic(True) if dynamicMuscleLabelsDict is not None else cmf.enableMuscleDynamic(False)
    
    trialManager = cmf.generate()

    #----cycles
    if eventType == "Gait":
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                               kinematicAcqs=trialManager.kinematic["Acqs"],
                                               kineticAcqs=trialManager.kinetic["Acqs"],
                                               emgAcqs=trialManager.emg["Acqs"],
                                               muscleGeometryAcqs=trialManager.muscleGeometry["Acqs"],
                                               muscleDynamicAcqs=trialManager.muscleDynamic["Acqs"])

    else:
        cycleBuilder = cycle.CyclesBuilder(spatioTemporalAcqs=trialManager.spatioTemporal["Acqs"],
                                           kinematicAcqs=trialManager.kinematic["Acqs"],
                                           kineticAcqs=trialManager.kinetic["Acqs"],
                                           emgAcqs=trialManager.emg["Acqs"],
                                           muscleGeometryAcqs=trialManager.muscleGeometry["Acqs"],
                                           muscleDynamicAcqs=trialManager.muscleDynamic["Acqs"])

    cyclefilter = cycle.CyclesFilter()
    cyclefilter.setBuilder(cycleBuilder)
    cycles = cyclefilter.build()

    if emgChannels is not None:
        emgLabelList = [label+"_Rectify_Env" for label in emgChannels]
    else:
        emgLabelList = None

    if eventType == "Gait":
        analysisBuilder = analysis.GaitAnalysisBuilder(cycles,
                                                       kinematicLabelsDict=kinematicLabelsDict,
                                                       kineticLabelsDict=kineticLabelsDict,
                                                       pointlabelSuffix=pointLabelSuffix,
                                                       emgLabelList=emgLabelList,
                                                       geometryMuscleLabelsDict=geometryMuscleLabelsDict,
                                                       dynamicMuscleLabelsDict=dynamicMuscleLabelsDict)
    else:
        analysisBuilder = analysis.AnalysisBuilder(cycles,
                                                   kinematicLabelsDict=kinematicLabelsDict,
                                                   kineticLabelsDict=kineticLabelsDict,
                                                   pointlabelSuffix=pointLabelSuffix,
                                                   emgLabelList=emgLabelList,
                                                   geometryMuscleLabelsDict=geometryMuscleLabelsDict,
                                                   dynamicMuscleLabelsDict=dynamicMuscleLabelsDict)

    analysisFilter = analysis.AnalysisFilter()
    analysisFilter.setInfo(subject=subjectInfo,
                           model=modelInfo, experimental=experimentalInfo)
    analysisFilter.setBuilder(analysisBuilder)
    analysisFilter.build()

    return analysisFilter.analysis


def exportAnalysis(analysisInstance:Analysis, DATA_PATH:str, name:str,
                   mode:str="Advanced"):
    """export an Analysis instance as excel spreadsheet.

    Args:
        analysisInstance (Analysis): Analysis instance.
        DATA_PATH (str): folder path
        name (str): name of your excel file.
        mode (str): spreadsheet mode . ("Advanced or Basic")

    Example:

    .. code-block:: python

        exportAnalysis(AnalysisInstance, "c:\\DATA\\","johnDoe")


    """

    exportFilter = exporter.XlsAnalysisExportFilter()
    exportFilter.setAnalysisInstance(analysisInstance)
    exportFilter.export(name, path=DATA_PATH, excelFormat="xls", mode=mode)


def automaticCPdeviations(
    DATA_PATH: str,
    analysis: Analysis,
    reference: str = "Nieuwenhuys2017",
    pointLabelSuffix: Optional[str] = None,
    filterTrue: bool = False,
    export: bool = True,
    outputname: str = "Nieuwenhuys2017",
    language: str = "-fr"):
    """
    Calculate and optionally export joint pattern deviations based on a specified reference.

    This function processes joint patterns using rules defined in an external Excel file,
    filters and retrieves pattern data, and can export the results to Excel files.

    Args:
        DATA_PATH (str): The path where the export files will be saved.
        analysis (Analysis): The `Analysis` object containing the necessary analysis data.
        reference (str, optional): Reference name for the joint pattern rules. Defaults to "Nieuwenhuys2017".
        pointLabelSuffix (Optional[str], optional): Suffix for the point labels. Defaults to None.
        filterTrue (bool, optional): If True, applies additional filtering to the patterns. Defaults to False.
        export (bool, optional): If True, exports the resulting data to Excel files. Defaults to True.
        outputname (str, optional): Base name for the output files. Defaults to "Nieuwenhuys2017".
        language (str, optional): Language specifier for the rules file, e.g., '-fr' for French. Defaults to "-fr".

    Returns:
        pd.DataFrame: A DataFrame containing the joint pattern deviations.

    Note:
        Requires external Excel files for rules, located in PYCGM2_SETTINGS_FOLDER.

    Example:
        >>> patterns = automaticCPdeviations("/path/to/data", analysisObj)
    """

    # Your function's code remains unchanged


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
