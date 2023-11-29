""" This module contains emg filters

check out the script : *\Tests\test_EMG.py* for examples

"""

import btk
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2 import enums
from pyCGM2.EMG.coactivationProcedures import CoActivationProcedure
from pyCGM2.EMG.discreteEmgProcedures import DiscreteEmgProcedure


import numpy as np
from pyCGM2.Signal import signal_processing
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.Processing import exporter
from pyCGM2.Processing.analysis import Analysis

from typing import List, Tuple, Dict, Optional
class BasicEmgProcessingFilter(object):
    """
    Filter for filtering EMG signals with a high-pass filter.

    This filter applies a high-pass Butterworth filter to EMG signals to remove low-frequency noise and baseline drift.

    Args:
        acq (btk.btkAcquisition): An acquisition instance containing EMG data.
        labels (List[str]): List of EMG channel labels to process.
    """

    def __init__(self,acq:btk.btkAcquisition, labels:List):
        """Initializes the BasicEmgProcessingFilter with acquisition data and EMG labels."""

        self.m_acq = acq
        self.m_labels = labels

    def setHighPassFrequencies(self,low:float,up:float):
        """Set the frequency boundaries of the EMG Butterworth high-pass filter.

        Args:
            low (float): Lower frequency boundary for the high-pass filter.
            up (float): Upper frequency boundary for the high-pass filter.
        """
        self.m_hpf_up = up
        self.m_hpf_low = low

    def run(self):
        """Run the high-pass filter on the specified EMG channels."""
        fa=self.m_acq.GetAnalogFrequency()
        for label in self.m_labels:
            values =  self.m_acq.GetAnalog(label).GetValues()
            # stop 50hz
            value50= signal_processing.remove50hz(values,fa)
            # high pass and compensation with mean
            hpLower = self.m_hpf_low
            hpUpper = self.m_hpf_up
            valuesHp =  signal_processing.highPass(value50,hpLower,hpUpper,fa)

            valuesHp = valuesHp - valuesHp.mean()

            btkTools.smartAppendAnalog(self.m_acq,label+"_HPF",valuesHp, desc= "high Pass filter" )

            # rectification
            btkTools.smartAppendAnalog(self.m_acq,label+"_Rectify",np.abs(valuesHp), desc= "rectify" )



class EmgEnvelopProcessingFilter(object):
    """
    Filter for processing EMG envelope from low-pass filter.

    This filter processes the rectified EMG signals by applying a low-pass filter to generate the EMG envelope.

    Args:
        acq (btk.btkAcquisition): BTK acquisition instance containing EMG data.
        labels (List[str]): List of EMG channel labels to process.
    """

    def __init__(self,acq:btk.btkAcquisition, labels:List):
        """Initializes the EmgEnvelopProcessingFilter with acquisition data and EMG labels."""


        self.m_acq = acq
        self.m_labels = [ label+"_Rectify" for label in labels]

    def setCutoffFrequency(self,fc:float):
        """Set the cut-off frequency for the low-pass filter.

        Args:
            fc (float): Cut-off frequency for the low-pass filter.
        """
        self.m_fc = fc


    def run(self):
        """Run the low-pass filter to generate the EMG envelope for each specified channel."""
        fa=self.m_acq.GetAnalogFrequency()
        for label in self.m_labels:
            values =  self.m_acq.GetAnalog(label).GetValues()
            valuesFilt = signal_processing.enveloppe(values, self.m_fc,fa)
            btkTools.smartAppendAnalog(self.m_acq,label+"_Env",valuesFilt, desc= "fc("+str(self.m_fc)+")")


class EmgNormalisationProcessingFilter(object):
    """
    Filter for normalizing EMG signals in amplitude.

    This filter normalizes the amplitude of EMG signals, typically after rectification and enveloping, using various normalization methods.

    Args:
        analysis (Analysis): A pyCGM2 analysis instance containing EMG data.
        label (str): EMG label for the signal to normalize.
        context (str): Event context (e.g., 'Left', 'Right').
    """

    def __init__(self,analysis:Analysis, label:str,context:str):
        """Initializes the EmgNormalisationProcessingFilter with an analysis instance, label, and context."""

        self.m_analysis = analysis
        self.m_label = label+"_Rectify_Env"
        self.m_context= context
        self.m_threshold = {}

        self.m_thresholdFromAnalysis = None

        self.__c3dProcess = False


    def setC3ds(self,datPath:str,c3ds:List,fileSuffix:str=None):
        """
        Set a list of C3D files for processing.

        Args:
            datPath (str): Folder data path containing the C3D files.
            c3ds (List[str]): List of C3D file names.
            fileSuffix (Optional[str]): Optional suffix added to C3D filenames during processing.
        """

        self.__c3dProcess = True
        self.m_c3dPath = datPath
        self.m_c3ds = c3ds
        self.m_fileSuffix=fileSuffix


    def setThresholdFromOtherAnalysis(self,analysis:Analysis):
        """
        Set another pyCGM2 analysis instance as the normalization denominator.

        Args:
            analysis (Analysis): A pyCGM2 analysis instance to use as the normalization reference.
        """
        self.m_thresholdFromAnalysis = analysis

    def setMaxMethod(self,EnumEmgNorm:enums.EmgAmplitudeNormalization, Value:float=None):
        """
        Set the normalization method for EMG signal amplitude.

        Args:
            EnumEmgNorm (enums.EmgAmplitudeNormalization): Enumeration specifying the normalization method.
            Value (Optional[float]): Optional value to force the normalization denominator.
        """

        if self.m_thresholdFromAnalysis is None:
            value = self.m_analysis.emgStats.data[self.m_label,self.m_context]
        else:
             value = self.m_thresholdFromAnalysis.emgStats.data[self.m_label,self.m_context]

        if EnumEmgNorm == enums.EmgAmplitudeNormalization.MaxMax:
            self.m_threshold = np.max(value["maxs"])

        elif EnumEmgNorm == enums.EmgAmplitudeNormalization.MeanMax:
            self.m_threshold = np.mean(value["maxs"])

        elif EnumEmgNorm == enums.EmgAmplitudeNormalization.MedianMax:
            self.m_threshold = np.median(value["maxs"])



    def processC3d(self):
        """
        Process all C3D files specified in the filter.

        Each C3D file is updated to include a new analog label with the suffix '_Norm'.
        """

        for c3d in self.m_c3ds:
            filenameOut  = c3d[:-4] + "_" + self.m_fileSuffix+".c3d" if self.m_fileSuffix is not None else c3d
            acq = btkTools.smartReader((self.m_c3dPath+c3d))

            values =  acq.GetAnalog(self.m_label).GetValues()
            valuesNorm = values / self.m_threshold

            btkTools.smartAppendAnalog(acq,self.m_label+"_Norm",valuesNorm, desc= "Normalization)")

            btkTools.smartWriter(acq, (self.m_c3dPath+filenameOut))


    def processAnalysis(self):
        """
        Process the pyCGM2 analysis instance for EMG normalization.

        New labels with the suffix '_Norm' are created in the 'emgStats.data' section of the pyCGM2 analysis instance.
        """

        for contextIt in ["Left","Right"]:
            if (self.m_label,contextIt) in self.m_analysis.emgStats.data:
                values = self.m_analysis.emgStats.data[self.m_label,contextIt]["values"]
                valuesNorm = []

                for val in values:
                    valuesNorm.append( val / self.m_threshold)

                self.m_analysis.emgStats.data[self.m_label+"_Norm",contextIt] = {
                        'mean':np.mean(valuesNorm,axis=0),
                        'median':np.median(valuesNorm,axis=0),
                        'std':np.std(valuesNorm,axis=0),
                        'values': valuesNorm}
            else:
                LOGGER.logger.warning("[pyCGM2] label [%s] - context [%s] dont find in the emgStats dictionary"%(self.m_label,contextIt))


    def run(self):
        """
        Run the EMG normalization filter.

        Processes either the specified C3D files or the pyCGM2 analysis instance, depending on the configuration.
        """

        if self.__c3dProcess:
            self.processC3d()
        self.processAnalysis()




class EmgCoActivationFilter(object):
    """
    Filter for computing co-activation index between two EMG signals.

    This filter computes the co-activation index between two specified EMG signals for a given context, using a specified co-activation procedure.

    Args:
        analysis (Analysis): A pyCGM2 analysis instance containing EMG data.
        context (str): Event context for the co-activation computation.
    """

    def __init__(self,analysis:Analysis,context:str):
        """Initializes the EmgCoActivationFilter with an analysis instance and context."""
        


        self.m_analysis = analysis
        self.m_context = context
        self.m_labelEMG1 = None
        self.m_labelEMG2 = None

    def setEMG1(self,label:str):
        """
        Set the label of the first EMG signal for co-activation computation.

        Args:
            label (str): The label of the first EMG signal.
        """
        self.m_labelEMG1 = label


    def setEMG2(self,label:str):
        """
        Set the label of the second EMG signal for co-activation computation.

        Args:
            label (str): The label of the first EMG signal.
        """

        self.m_labelEMG2 = label


    def setCoactivationMethod(self, concreteCA:CoActivationProcedure):
        """
        Set the co-activation procedure for computing the co-activation index.

        Args:
            concreteCA (CoActivationProcedure): An instance of a CoActivationProcedure subclass for computing the co-activation index.
        """

        self.m_concreteCA = concreteCA

    def run(self):
        """
        Run the co-activation filter.

        Computes the co-activation index between the two specified EMG signals and updates the 'coactivation' section of the pyCGM2 Analysis instance with descriptive statistics of the results.
        """
        emg1v = self.m_analysis.emgStats.data[self.m_labelEMG1+"_Rectify_Env_Norm",self.m_context]["values"]
        emg2v = self.m_analysis.emgStats.data[self.m_labelEMG2+"_Rectify_Env_Norm",self.m_context]["values"]

        res = self.m_concreteCA.run(emg1v,emg2v)

        resDict = {"mean":np.mean(res) ,
                   "median":np.median(res),
                   "std":np.std(res),
                   "values":res}

        self.m_analysis.setCoactivation(self.m_labelEMG1,self.m_labelEMG2,self.m_context,resDict)

class DiscreteEMGFilter(object):
    """
    Filter for handling discrete EMG procedures.

    This filter is designed to apply a specified discrete EMG procedure to an analysis instance. It generates a Pandas DataFrame as output, containing the results of the discrete EMG analysis.

    Args:
        discreteEMGProcedure (DiscreteEmgProcedure): An instance of a discrete EMG procedure.
        analysis (Analysis): A pyCGM2 analysis instance containing EMG data.
        emgLabels (List[str]): List of EMG labels.
        emgMuscles (List[str]): Corresponding muscle names for EMG labels.
        emgContexts (List[str]): Context side ('Left', 'Right') for each EMG label.
        subjInfo (Optional[Dict]): Optional dictionary describing subject information.
        condExpInfo (Optional[Dict]): Optional dictionary describing experimental conditions.
    """

    def __init__(self, discreteEMGProcedure:DiscreteEmgProcedure, analysis:Analysis, 
                 emgLabels:List, emgMuscles:List, emgContexts:List, subjInfo:Dict=None, condExpInfo:Dict=None):
        """Initializes the DiscreteEMGFilter with a procedure, analysis instance, EMG labels, muscles, contexts, and optional subject/experiment information."""
        
        self.m_procedure = discreteEMGProcedure
        self.m_analysis = analysis
        self.dataframe = None

        self.m_subjInfo = subjInfo
        self.m_condExpInfo = condExpInfo

        self.m_emgMuscles = emgMuscles
        self.m_emgLabels = emgLabels
        self.m_emgContexts = emgContexts

    def setSubjInfo(self, subjInfo:Dict):
        """
        Set subject information for inclusion in the output DataFrame.

        Args:
            subjInfo (Dict): Dictionary describing subject information. Items from this dictionary will be added to the generated DataFrame.
        """

        self.m_subjInfo = subjInfo

    def setCondExpInf(self, condExpInfo:Dict):
        """
        Set experimental condition information for inclusion in the output DataFrame.

        Args:
            condExpInfo (Dict): Dictionary describing experimental conditions. Items from this dictionary will be added to the generated DataFrame.
        """
        self.m_condExpInfo = condExpInfo

    def getOutput(self)->pd.DataFrame:
        """
        Run the discrete EMG procedure and get the output DataFrame.

        Processes the specified EMG data using the discrete EMG procedure and generates a Pandas DataFrame containing the results.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the discrete EMG analysis.
        """
        self.dataframe = self.m_procedure.detect(
            self.m_analysis, self.m_emgLabels, self.m_emgMuscles, self.m_emgContexts)

        # add infos

        if self.m_subjInfo is not None:
            for key, value in self.m_subjInfo.items():
                exporter.isColumnNameExist(self.dataframe, key)
                self.dataframe[key] = value

        if self.m_condExpInfo is not None:
            for key, value in self.m_condExpInfo.items():
                exporter.isColumnNameExist(self.dataframe, key)
                self.dataframe[key] = value

        return self.dataframe
