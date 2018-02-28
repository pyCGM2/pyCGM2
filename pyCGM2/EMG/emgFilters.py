# -*- coding: utf-8 -*-
import numpy as np
from pyCGM2.Signal import signal_processing
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.EMG import coactivation

import btk

class BasicEmgProcessingFilter(object):
    """

    """

    def __init__(self,acq, labels):

        self.m_acq = acq
        self.m_labels = labels

    def setHighPassFrequencies(self,low,up):
        self.m_hpf_up = up
        self.m_hpf_low = low

    def run(self):

        fa=self.m_acq.GetAnalogFrequency()
        for analog in btk.Iterate(self.m_acq.GetAnalogs()):
            for label in self.m_labels:
                if analog.GetLabel()==label :

                    values =  analog.GetValues()

                    # stop 50hz
                    value50= signal_processing.remove50hz(values,fa)

                    # high pass and compensation with mean
                    hpLower = self.m_hpf_low
                    hpUpper = self.m_hpf_up


                    valuesHp =  signal_processing.highPass(value50,hpLower,hpUpper,fa)

                    btkTools.smartAppendAnalog(self.m_acq,label+"_HPF",valuesHp, desc= "high Pass filter" )

                    # rectification
                    btkTools.smartAppendAnalog(self.m_acq,label+"_Rectify",np.abs(valuesHp), desc= "rectify" )

class EmgEnvelopProcessingFilter(object):
    """

    """

    def __init__(self,acq, labels):

        self.m_acq = acq
        self.m_labels = [ label+"_Rectify" for label in labels]

    def setCutoffFrequency(self,fc):
        self.m_fc = fc


    def run(self):
        fa=self.m_acq.GetAnalogFrequency()
        for analog in btk.Iterate(self.m_acq.GetAnalogs()):
            for label in self.m_labels:
                if analog.GetLabel()==label :

                    values =  analog.GetValues()
                    valuesFilt = signal_processing.enveloppe(values, self.m_fc,fa)

                    btkTools.smartAppendAnalog(self.m_acq,label+"_Env",valuesFilt, desc= "fc("+str(self.m_fc)+")")

class EmgNormalisationProcessingFilter(object):
    """

    """

    def __init__(self,analysis, label,context):

        self.m_analysis = analysis
        self.m_label = label+"_Rectify_Env"
        self.m_context= context
        self.m_threshold = dict()


    def setMaxMethod(self,EnumEmgNorm, Value=None):


        if EnumEmgNorm == enums.EmgAmplitudeNormalization.MaxMax:
            self.m_threshold = np.max(self.m_analysis.emgStats.data[self.m_label,self.m_context]["maxs"])

        elif EnumEmgNorm == enums.EmgAmplitudeNormalization.MeanMax:
            self.m_threshold = np.mean(self.m_analysis.emgStats.data[self.m_label,self.m_context]["maxs"])

        elif EnumEmgNorm == enums.EmgAmplitudeNormalization.MedianMax:
            self.m_threshold = np.median(self.m_analysis.emgStats.data[self.m_label,self.m_context]["maxs"])

        elif EnumEmgNorm == enums.EmgAmplitudeNormalization.Threshold:

            if Value is None:
                raise Exception ("[pyCGM2] : You need to input a Threhsold value")

            self.m_threshold = Value

    def processC3d(self,c3ds,fileSuffix=None):
        for c3d in c3ds:
            if fileSuffix is not None:
                fileSuffix

            filenameOut  = c3d[:-4] + "_" + fileSuffix+".c3d" if fileSuffix is not None else c3d
            acq = btkTools.smartReader(str(c3d))

            values =  acq.GetAnalog(self.m_label).GetValues()
            valuesNorm = values / self.m_threshold

            btkTools.smartAppendAnalog(acq,self.m_label+"_Norm",valuesNorm, desc= "Normalization)")

            btkTools.smartWriter(acq, str(filenameOut))


    def run(self):

        values = self.m_analysis.emgStats.data[self.m_label,self.m_context]["values"]

        n = len(values)
        valuesNorm = list()

        i=0
        for val in values:
            print i
            valuesNorm.append( val / self.m_threshold)
            i+=1

        valuesNormArray = np.asarray(valuesNorm).reshape((101,n))


        self.m_analysis.emgStats.data[self.m_label+"_Norm",self.m_context] = {
                'mean':np.mean(valuesNormArray,axis=1).reshape((101,1)),
                'median':np.median(valuesNormArray,axis=1).reshape((101,1)),
                'std':np.std(valuesNormArray,axis=1).reshape((101,1)),
                'values': valuesNorm}


class EmgCoActivationFilter(object):
    """

    """

    def __init__(self,analysis,context):

        self.m_analysis = analysis
        self.m_context = context
        self.m_labelEMG1 = None
        self.m_labelEMG2 = None

    def setEMG1(self,label):
        self.m_labelEMG1 = label


    def setEMG2(self,label):
        self.m_labelEMG2 = label


    def setCoactivationMethod(self, concreteCA):
        self.m_concreteCA = concreteCA

    def run(self):
        emg1v = self.m_analysis.emgStats.data[self.m_labelEMG1+"_Rectify_Env_Norm",self.m_context]["values"]
        emg2v = self.m_analysis.emgStats.data[self.m_labelEMG2+"_Rectify_Env_Norm",self.m_context]["values"]

        res = self.m_concreteCA.run(emg1v,emg2v)

        resDict = {"mean":np.mean(res) ,
                   "median":np.median(res),
                   "std":np.std(res),
                   "values":res}

        self.m_analysis.setCoactivation(self.m_labelEMG1,self.m_labelEMG2,self.m_context,resDict)
