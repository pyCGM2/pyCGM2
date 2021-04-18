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
                    pstfilenames=None,kinematicfilenames=None,kineticfilenames=None,emgfilenames=None,

                    ):

    """This function normalise data in time and return an **Analysis Instance** ie a nested dictionnary container containing
       spatiotemporal parameters, normalized kinematics, normalized kinetics and normalized EMG envelops from a list of c3d files.

       By default: the function calls :

        * kinematic and kinetic ouputs of the CGM
        * emg channels names Voltage.EMG1 to Voltage.EMG16

       You can also compute spatiotemporal parameters, normalized kinematics, normalized kinetics and normalized EMG envelops
       from different set of c3d files. For that  use the named arguments :

         * pstfilenames
         * kinematicfilenames
         * kineticfilenames
         * emgfilenames



    Args:
        DATA_PATH (str): folder path [REQUIRED]
        filenames (list): list of c3d files to normalize [REQUIRED]
        type (str): event type (choice : "Gait" or "unknown").
        kinematicLabelsDict (dict): dictionnary containing kinematic data to normalize.
        kineticLabelsDict (dict): dictionnary containing kinetic data to normalize.
        emgChannels (list): list of emg channel
        pointLabelSuffix (str): suffix associated to pont output
        btkAcqs (list of btk.Acquisition): btkAcq instances to process instead of calling c3d file.
        subjectInfo (dict): dictionnary with metadata information about the subject.
        experimentalInfo (dict): dictionnary with metadata information about the expreiment.
        modelInfo (dict): dictionnary with metadata information about the model.
        pstfilenames (list): list of c3d files used for computing spatiotemporal parameters
        kinematicfilenames (list): list of c3d files used to normalize kinematic data
        kineticfilenames (list): list of c3d files used to normalize kinetic data
        emgfilenames (list): list of c3d files used to normalize emg data

    Returns:
        pyCGM2.Processing.analysis.Analysis: analysis instance


    Examples:

        >>> analysisInstance2 = analysis.makeAnalysis(DATA_PATH,
              [file1.c3d,"file2.c3d"],
              type="Gait",
              kinematicLabelsDict = {"Left": ["LHipAngles,LKneeAngles"], "Right": ["RHipAngles,RKneeAngles"]},
              kineticLabelsDict = {"Left": ["LHipMoment,LKneePower"], "Right": ["RHipMoment,RKneeMoment"],
              emgChannels = ["Voltage.EMG1","Voltage.EMG2","Voltage.EMG3"],
              pointLabelSuffix="cgm1",
              subjectInfo = {"Name":"Doe","Firstname":"John"},
              experimentalInfo = {"Barefoot":"No"},
              modelInfo = {"Model":"CGM1"})




    """


    if filenames == [] or filenames is None: filenames=None

    #---- c3d manager

    if btkAcqs is not None:
        c3dmanagerProcedure = c3dManager.UniqueBtkAcqSetProcedure(DATA_PATH,filenames,acqs=btkAcqs)

    else:
        iPstFilenames =  filenames if pstfilenames is None else pstfilenames
        iKinematicFilenames =  filenames if kinematicfilenames is None else kinematicfilenames
        iKineticFilenames =  filenames if kineticfilenames is None else kineticfilenames
        iEmgFilenames =  filenames if emgfilenames is None else emgfilenames

        c3dmanagerProcedure = c3dManager.DistinctC3dSetProcedure(DATA_PATH, iPstFilenames, iKinematicFilenames, iKineticFilenames, iEmgFilenames)

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
    """export an Analysis instance as excel spreadsheet.

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): Analysis instance.
        DATA_PATH (str):folder path
        name (str): name of your excel file.
        mode (str): spreadsheet mode . ("Advanced or basic")

    """



    exportFilter = exporter.XlsAnalysisExportFilter()
    exportFilter.setAnalysisInstance(analysisInstance)
    exportFilter.export(name, path=DATA_PATH,excelFormat = "xls",mode = mode)


def automaticCPdeviations(DATA_PATH,analysis,pointLabelSuffix=None,filterTrue=False, export=True, outputname ="Nieuwenhuys2017" ):


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
