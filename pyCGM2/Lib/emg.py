# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Functions
#APIDOC["Draft"]=False
#--end--
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2.EMG import emgManager
from pyCGM2 import enums

def loadEmg(DATA_PATH):
    """
    Load and manage emg settings

    Args:
        DATA_PATH (str): folder path.

    Returns:
        emgManager (pyCGM2.EMG.EmgManager): an emg manager instance
    """

    return emgManager.EmgManager(DATA_PATH)


def processEMG(DATA_PATH, gaitTrials, emgChannels,
                highPassFrequencies=[20,200],envelopFrequency=6.0, fileSuffix=None,outDataPath=None):
    """ basic filtering of EMG from c3d files .

    Args:
        DATA_PATH (str): folder path.
        gaitTrials (str): list of c3d files.
        emgChannels (list): list or emg channel
        highPassFrequencies (list)[20,200]: boundaries of the bandpass filter
        envelopFrequency (float)[6.0]: cut-off frequency of low pass emg
        fileSuffix (str)[None]: add a suffix to the exported c3d files
        outDataPath (str)[None]: path to place the exported c3d files.

    Examples:

    .. code-block:: python

        emg.processEMG(DATA_PATH, ["file1.c3d","file2.c3d"], ["Voltage.EMG1","Voltage.EMG2"])

    The code loads 2 c3d files, then processes the analog channel name `Voltage.EMG1`
    and `Voltage.EMG2`

    """

    if fileSuffix is None: fileSuffix = ""

    for gaitTrial in gaitTrials:
        acq = btkTools.smartReader(DATA_PATH + gaitTrial)

        for channel in emgChannels:
            if not btkTools.isAnalogExist(acq,channel):
                raise Exception("channel [%s] not detected in the c3d [%s]" % (channel, gaitTrial))


        bf = emgFilters.BasicEmgProcessingFilter(acq, emgChannels)
        bf.setHighPassFrequencies(highPassFrequencies[0], highPassFrequencies[1])
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
        envf.setCutoffFrequency(envelopFrequency)
        envf.run()

        outFilename = gaitTrial if fileSuffix=="" else gaitTrial[0:gaitTrial.rfind(".")]+"_"+fileSuffix+".c3d"

        if outDataPath is None:
            btkTools.smartWriter(acq,DATA_PATH+outFilename)
        else:
            btkTools.smartWriter(acq,outDataPath+outFilename)


def normalizedEMG(DATA_PATH,analysis, method="MeanMax", fromOtherAnalysis=None, mvcSettings=None,**kwargs):
    """
    Emg normalisation in amplitude.

    This function update the analysis instance with normalized emg signal in amplitude

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): an analysis Instance
        DATA_PATH (str): folder path
        method (str)["MeanMax"]: normalisation method (choice : MeanMax, MaxMax, MedianMax ).
        fromOtherAnalysis (pyCGM2.Processing.analysis.Analysis)[None]: normalise in amplitude from another analysis instance.
        mvcSettings (dict)[None]: mvc settings.

    Keyword Arguments:
        forceEmgManager (pyCGM2.Emg.EmgManager)[None]: force the use of a specific emgManager instance.



    Examples:

    .. code-block:: python

        emg.normalizedEMG(emgAnalysisInstance,
        .................method="MeanMax",
        .................fromOtherAnalysis=emgAnalysisInstancePreBloc)


    The code normalized emg channels of the current analysis instance `emgAnalysisInstance`
    from the mean maximum values of an other analysis instance `emgAnalysisInstancePreBloc`

    """

    if "forceEmgManager" in kwargs:
        emg = kwargs["forceEmgManager"]
    else:
        emg = emgManager.EmgManager(DATA_PATH)

    emgChannels = emg.getChannels()
    contexts = emg.getSides()

    rows = list()
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
        rows.append([envnf.m_label, envnf.m_threshold])
        del envnf

    df = pd.DataFrame(rows, columns=["Label", "MvcThreshold"])
    return df


def processEMG_fromBtkAcq(acq, emgChannels, highPassFrequencies=[20,200],envelopFrequency=6.0):
    """ Process EMG from a btk.acquisition

    Args:
        acq (btk.Acquisition): an acquisition instance
        emgChannels (list): emg channels ( ie analog labels )
        highPassFrequencies (list,Optional[20,200]): high pass frequencies
        envelopFrequency (float,Optional[6.0]): low pass filter frequency

    Examples:

    .. code-block:: python

        emg.processEMG_fromBtkAcq(acq,
        .................["Voltage.EMG1","Voltage.EMG2"])

    """

    bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
    bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
    bf.run()

    envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
    envf.setCutoffFrequency(envelopFrequency)
    envf.run()

    return acq
