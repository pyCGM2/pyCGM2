import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2.EMG import emgManager
from pyCGM2 import enums
from pyCGM2.Processing.analysis import Analysis

from typing import List, Tuple, Dict, Optional,Union
import btk

def loadEmg(DATA_PATH:str):
    """
    Load EMG data and create an EMG manager instance.

    This function initializes an EmgManager instance using a specified data path.

    Args:
        DATA_PATH (str): The folder path where EMG data and settings are located.

    Returns:
        pyCGM2.EMG.EmgManager: An initialized EmgManager instance for managing EMG data.

    Example:
        >>> emg_manager = loadEmg("/path/to/data")
    """

    return emgManager.EmgManager(DATA_PATH)


def processEMG(DATA_PATH:str, 
               gaitTrials:List, 
               emgChannels:List,
               highPassFrequencies:List=[20,200],
               envelopFrequency:float=6.0, 
               fileSuffix:Optional[str]=None,
               outDataPath:Optional[str]=None):
    """
    Process and filter EMG data from c3d files.

    This function applies basic EMG filtering to specified c3d files. It supports high-pass
    and low-pass filtering and allows exporting the processed data.

    Args:
        DATA_PATH (str): The folder path containing c3d files.
        gaitTrials (List[str]): A list of c3d file names to process.
        emgChannels (List[str]): A list of EMG channel names to be processed.
        highPassFrequencies (List[int]): The boundaries of the bandpass filter. Defaults to [20, 200].
        envelopFrequency (float): The cut-off frequency for the low pass filter. Defaults to 6.0.
        fileSuffix (Optional[str]): A suffix for the exported c3d files. Defaults to None.
        outDataPath (Optional[str]): The path to save the exported c3d files. Defaults to None.

    Example:
        >>> processEMG("/data/path", ["gaitTrial1.c3d", "gaitTrial2.c3d"], ["Voltage.EMG1", "Voltage.EMG2"])

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


def normalizedEMG(DATA_PATH:str,
                  analysis:Analysis, 
                  method:str="MeanMax", fromOtherAnalysis:Optional[Analysis]=None, 
                  mvcSettings:Optional[Dict]=None,
                  **kwargs):
    """
    Normalize EMG data in amplitude and update the analysis instance.

    This function normalizes the amplitude of EMG signals in an analysis instance, using 
    different normalization methods. It can also normalize based on another analysis instance 
    or specified MVC settings.

    Args:
        DATA_PATH (str): Folder path for EMG data.
        analysis (Analysis): An Analysis instance to be updated with normalized EMG data.
        method (str): Normalization method. Choices include 'MeanMax', 'MaxMax', and 'MedianMax'. Defaults to 'MeanMax'.
        fromOtherAnalysis (Optional[Analysis]): Use another Analysis instance for normalization. Defaults to None.
        mvcSettings (Optional[Dict]): MVC settings for normalization. Defaults to None.

    Keyword Args:
        forceEmgManager (pyCGM2.EMG.EmgManager): Force the use of a specific EmgManager instance.

    Returns:
        pd.DataFrame: A DataFrame with labels and MVC thresholds.

    Example:
       
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

    rows = []
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


def processEMG_fromBtkAcq(acq:btk.btkAcquisition, emgChannels:List[str], 
                          highPassFrequencies:List[int]=[20,200],
                          envelopFrequency:float=6.0):
    """
    Process EMG data from a BTK acquisition instance.

    This function applies basic and envelop processing filters to EMG data within a BTK acquisition instance.

    Args:
        acq (btk.btkAcquisition): A BTK acquisition instance containing EMG data.
        emgChannels (List[str]): A list of EMG channel names to process.
        highPassFrequencies (List[int]): The high-pass filter frequencies. Defaults to [20, 200].
        envelopFrequency (float): The cut-off frequency for the low-pass filter. Defaults to 6.0.

    Returns:
        btk.btkAcquisition: The processed BTK acquisition instance.

    Example:
        >>> processed_acq = processEMG_fromBtkAcq(acq, ["Voltage.EMG1", "Voltage.EMG2"])
    """

    bf = emgFilters.BasicEmgProcessingFilter(acq,emgChannels)
    bf.setHighPassFrequencies(highPassFrequencies[0],highPassFrequencies[1])
    bf.run()

    envf = emgFilters.EmgEnvelopProcessingFilter(acq,emgChannels)
    envf.setCutoffFrequency(envelopFrequency)
    envf.run()

    return acq
