# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/EMG
#APIDOC["Draft"]=False
#--end--

""" This module contains emg filters handling emg procedures

check out the script : *\Tests\test_EMG.py* for examples

"""


import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np
from pyCGM2.Signal import signal_processing
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.Processing import exporter

class BasicEmgProcessingFilter(object):
    """
    Filter for filtering emg signals with a high pass filter

    Args:
        acq (Btk.Acquisition): btk acquisition instance
        labels (list): emg labels.
    """

    def __init__(self,acq, labels):

        self.m_acq = acq
        self.m_labels = labels

    def setHighPassFrequencies(self,low,up):
        """Set the frequency boudaries of your emg Butterworth high-pass filter.

        Args:
            low (float): lower frequency
            up (float): upper frequency

        """
        self.m_hpf_up = up
        self.m_hpf_low = low

    def run(self):
        """Run the filter  """
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
    Filter for processing emg envelop from low-pass filter
    Args:
        acq (Btk.Acquisition): btk acquisition instance
        labels (list): emg labels.
    """

    def __init__(self,acq, labels):

        self.m_acq = acq
        self.m_labels = [ label+"_Rectify" for label in labels]

    def setCutoffFrequency(self,fc):
        """Set the cut-off frequency.

        Args:
            fc (float): cut-off frequency
        """
        self.m_fc = fc


    def run(self):
        """Run the filter  """
        fa=self.m_acq.GetAnalogFrequency()
        for label in self.m_labels:
            values =  self.m_acq.GetAnalog(label).GetValues()
            valuesFilt = signal_processing.enveloppe(values, self.m_fc,fa)
            btkTools.smartAppendAnalog(self.m_acq,label+"_Env",valuesFilt, desc= "fc("+str(self.m_fc)+")")


        # for analog in btk.Iterate(self.m_acq.GetAnalogs()):
        #
        #     for label in self.m_labels:
        #         if analog.GetLabel()==label :
        #
        #             values =  analog.GetValues()
        #             valuesFilt = signal_processing.enveloppe(values, self.m_fc,fa)
        #
        #             btkTools.smartAppendAnalog(self.m_acq,label+"_Env",valuesFilt, desc= "fc("+str(self.m_fc)+")")

class EmgNormalisationProcessingFilter(object):
    """
    Filter for normalizing emg signals in amplitude

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): A pycgm2 analysis instance
        label (str): emg label.
        context (str): Event context.
    """

    def __init__(self,analysis, label,context):


        self.m_analysis = analysis
        self.m_label = label+"_Rectify_Env"
        self.m_context= context
        self.m_threshold = dict()

        self.m_thresholdFromAnalysis = None

        self.__c3dProcess = False


    def setC3ds(self,datPath,c3ds,fileSuffix=None):
        """Set a list of c3d.

        Args:
            datPath (str): Folder data path
            c3ds (list): the names of c3d
            fileSuffix (str,optional): suffix added to c3d filenames
        """

        self.__c3dProcess = True
        self.m_c3dPath = datPath
        self.m_c3ds = c3ds
        self.m_fileSuffix=fileSuffix


    def setThresholdFromOtherAnalysis(self,analysis):
        """Set an other pyCGM2 analysis instance as normalisation denominator

        Args:
            analysis (pyCGM2.Processing.analysis.Analysis): A pycgm2 analysis instance

        """
        self.m_thresholdFromAnalysis = analysis

    def setMaxMethod(self,EnumEmgNorm, Value=None):
        """set a normalisation method

        Args:
            EnumEmgNorm (pyCGM2.Enums): method
            Value (float,Optional): force the denominator value

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
        """ Process all c3d filenames

        Each c3d are updated and include a new analog label with the suffix *_Norm*

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
        Process the pyCGM2 analysis instance

        New labels with the suffix *_Norm* is created in the section emgStats.data of the pyCGM2 analysis instance
        """

        for contextIt in ["Left","Right"]:
            if (self.m_label,contextIt) in self.m_analysis.emgStats.data:
                values = self.m_analysis.emgStats.data[self.m_label,contextIt]["values"]
                valuesNorm = list()

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
        """ run the filter"""

        if self.__c3dProcess:
            self.processC3d()
        self.processAnalysis()




class EmgCoActivationFilter(object):
    """
    Filter for computing coactivation index

    Args:
        analysis (pyCGM2.Processing.analysis.Analysis): A pycgm2 analysis instance
        context (str): event context
    """

    def __init__(self,analysis,context):


        self.m_analysis = analysis
        self.m_context = context
        self.m_labelEMG1 = None
        self.m_labelEMG2 = None

    def setEMG1(self,label):
        """set the label of the first emg signal

        Args:
            label (str): emg label
        """
        self.m_labelEMG1 = label


    def setEMG2(self,label):
        """set the label of the second emg signal

        Args:
            label (str): emg label
        """

        self.m_labelEMG2 = label


    def setCoactivationMethod(self, concreteCA):
        """set the coactivation procedure

        Args:
            concreteCA (pyCGM2.EMG.Coactivation): Coactivation procedure instance

        """

        self.m_concreteCA = concreteCA

    def run(self):
        """ run ethe filter

        The coactivation section of the pyCGM2-Analysis instance is updated with descriptive statistics
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
    Filter for handing procedure.

    the goal of this filter is to return a Pandas dataframe.

    Args:
        discreteEMGProcedure (pyCGM2.EMG.discreteEmgProcedures.procedure): a discrete emg procedure instance.
        analysis (pyCGM2.Processing.analysis.Analysis): A pycgm2 analysis instance
        emgLabels (list): emg labels
        emgMuscles (list): muscle matching emg labels
        emgContexts (list):  side of each emg labels
        subjInfo (dict,Optional[None]): dictionary decribing the subject. Items will be added to the generated pandas dataframe
        condExpInfo (dict,Optional[None]): dictionary decribing the experiment conditions. Items will be added to the generated pandas dataframe

    """

    def __init__(self, discreteEMGProcedure, analysis, emgLabels, emgMuscles, emgContexts, subjInfo=None, condExpInfo=None):

        self.m_procedure = discreteEMGProcedure
        self.m_analysis = analysis
        self.dataframe = None

        self.m_subjInfo = subjInfo
        self.m_condExpInfo = condExpInfo

        self.m_emgMuscles = emgMuscles
        self.m_emgLabels = emgLabels
        self.m_emgContexts = emgContexts

    def setSubjInfo(self, subjInfo):
        """ set subject info

        Args:
            subjInfo (dict): dictionary decribing the subject. Items will be added to the generated pandas dataframe

        """

        self.m_subjInfo = subjInfo

    def setCondExpInf(self, condExpInfo):
        """ set experiment condition info

        Args:
            condExpInfo (dict): dictionary decribing the experiment conditions. Items will be added to the generated pandas dataframe

        """
        self.m_condExpInfo = condExpInfo

    def getOutput(self):
        """run the procedure and get outputs

        Returns:
            pandas.Dataframe: DataFrame

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
