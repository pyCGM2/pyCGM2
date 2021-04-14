# -*- coding: utf-8 -*-
#import ipdb
import pyCGM2; LOGGER = pyCGM2.LOGGER
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters
from pyCGM2 import enums




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
