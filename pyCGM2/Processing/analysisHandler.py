# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

import pyCGM2; LOGGER = pyCGM2.LOGGER


def getPhases(dataStats,context="Both"):
    """get phases

    Args:
        dataStats (pyCGM2.Processing.analysis.Analysis.kinematicStats.data): `data` level
        of the `kinematicStats` or `kineticStats` attributes of an `Analysis` instance
        context (str): event context

    """
    #phases
    phases = dict()

    # Left
    if context == "Both" or context=="Left":
        phases["Left","ST"] = [0,int(dataStats.pst["stancePhase","Left"]["mean"])]
        phases["Left","SW"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                            100]

        phases["Left","DS1"] = [0,
                                        int(dataStats.pst["doubleStance1","Left"]["mean"])]

        phases["Left","SS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]) ]
        phases["Left","eSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"]),
                                        int(dataStats.pst["doubleStance1","Left"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Left"]["mean"])]
        phases["Left","mSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Left"]["mean"]),
                                        int(dataStats.pst["doubleStance1","Left"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Left"]["mean"])]
        phases["Left","lSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"])]

        phases["Left","DS2"] = [int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"])]

        phases["Left","eSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

        phases["Left","mSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

        phases["Left","lSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                        100]

    if context == "Both" or context=="Right":
        # Right
        phases["Right","ST"] = [0,int(dataStats.pst["stancePhase","Right"]["mean"])]
        phases["Right","SW"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                        100]

        phases["Right","DS1"] = [0,
                                        int(dataStats.pst["doubleStance1","Right"]["mean"])]
        phases["Right","SS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]) ]

        phases["Right","eSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"]),
                                        int(dataStats.pst["doubleStance1","Right"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Right"]["mean"])]
        phases["Right","mSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Right"]["mean"]),
                                        int(dataStats.pst["doubleStance1","Right"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Right"]["mean"])]
        phases["Right","lSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"])]

        phases["Right","DS2"] = [int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"])]

        phases["Right","eSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

        phases["Right","mSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

        phases["Right","lSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                        100]

    return phases


def getNumberOfCycle(analysisInstance,label,context):
    """return the number of cycle from a given ouput and its event context.

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an analysis instance
        label (str): givel ouput label (eg LKneeAngles)
        context (str): event context

    """

    statSection = getAnalysisSection(analysisInstance,label,context)
    values = statSection.data[label,context]["values"]
    n_cycle = len(values)

    return n_cycle

def getValues(analysisInstance,label,context):
    """return  the values from a given ouput and its event context.

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an analysis instance
        label (str): givel ouput label (eg LKneeAngles)
        context (str): event context

    """

    statSection = getAnalysisSection(analysisInstance,label,context)
    values = statSection.data[label,context]["values"]

    return values



def isKeyExist(analysisInstance,label,context,exceptionMode = False):
    """check if the combinaison label/context exist in the analysis instance

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an analysis instance
        label (str): givel ouput label (eg LKneeAngles)
        context (str): event context
        exceptionMode (bool): enable exception raising

    """

    flag = False
    for item in  [analysisInstance.kinematicStats.data,analysisInstance.kineticStats.data,analysisInstance.emgStats.data]:
        if (label,context) in item.keys():
            flag = True
            break

    if not flag:
        if exceptionMode:
            raise Exception ("[pyCGM2] label [%s] - context [%s] doesn t find in the analysis instance"%(label,context))
        else:
            LOGGER.logger.error("[pyCGM2] label [%s] - context [%s] doesn t find in the analysis instance"%(label,context))

    return flag

def getAnalysisSection(analysisInstance,label,context):
    """return the section ( ie kinematic, kinetic or emg) of a given label and its context

    Args:
        analysisInstance (pyCGM2.Processing.analysis.Analysis): an analysis instance
        label (str): givel ouput label (eg LKneeAngles)
        context (str): event context

    """

    if (label,context) in analysisInstance.kinematicStats.data.keys():
        return analysisInstance.kinematicStats
    elif (label,context) in analysisInstance.kineticStats.data.keys():
        return analysisInstance.kineticStats
    elif (label,context) in analysisInstance.emgStats.data.keys():
        return analysisInstance.emgStats
    else:
        raise Exception ("[pyCGM2] label [%s] - context [%s] doesn t find in the analysis instance"%(label,context))
