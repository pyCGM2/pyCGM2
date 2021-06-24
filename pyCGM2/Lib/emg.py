# -*- coding: utf-8 -*-
#import ipdb
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters,emgManager
from pyCGM2 import enums

def loadEmg(DATA_PATH):

    return emgManager.EmgManager(DATA_PATH)


def processEMG(DATA_PATH, gaitTrials, emgChannels,
                highPassFrequencies=[20,200],envelopFrequency=6.0, fileSuffix=None,outDataPath=None):
    """ basic filtering of EMG from c3d files .

    Args:
        DATA_PATH (str): folder path.
        gaitTrials (str): list of c3d files.
        emgChannels (list): list or emg channel

    Keyword Args:
        highPassFrequencies (list)[20,200]: boundaries of the bandpass filter
        envelopFrequency (float)[6.0]: cut-off frequency of low pass emg
        fileSuffix (str)[None]: add a suffix to the exported c3d files
        outDataPath (str)[None]: path to place the exported c3d files.

    Examples

    ```python
    emg.processEMG(DATA_PATH, ["file1.c3d","file2.c3d"], ["Voltage.EMG1","Voltage.EMG2"])
    ```

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


def normalizedEMG(DATA_PATH,analysis, method="MeanMax", fromOtherAnalysis=None, mvcSettings=None):
    """
    Emg normalisation in amplitude.

    This function update the analysis instance with normalized emg signal in amplitude

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): an analysis Instance
        DATA_PATH (str): folder path

    Keyword Args:
        method (str)["MeanMax"]: normalisation method (choice : MeanMax, MaxMax, MedianMax ).
        fromOtherAnalysis (pyCGM2.Processing.analysis.Analysis)[None]: normalise in amplitude from another analysis instance.
        mvcSettings (dict)[None]: mvc settings.



    Examples:

    ```python
    emg.normalizedEMG(emgAnalysisInstance,
    .................["Voltage.EMG1","Voltage.EMG2"],
    .................["Left","Right"],
    .................method="MeanMax",
    .................fromOtherAnalysis=emgAnalysisInstancePreBloc)
    ```

    """

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

    bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
    bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
    bf.run()

    envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
    envf.setCutoffFrequency(envelopFrequency)
    envf.run()

    return acq
