# -*- coding: utf-8 -*-
import logging
import numpy as np
from pyCGM2.Signal import signal_processing
from pyCGM2.Tools import btkTools
from pyCGM2 import enums


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

    """

    def __init__(self,acq, labels):

        self.m_acq = acq
        self.m_labels = [ label+"_Rectify" for label in labels]

    def setCutoffFrequency(self,fc):
        self.m_fc = fc


    def run(self):
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

    """

    def __init__(self,analysis, label,context):

        self.m_analysis = analysis
        self.m_label = label+"_Rectify_Env"
        self.m_context= context
        self.m_threshold = dict()

        self.m_thresholdFromAnalysis = None

        self.__c3dProcess = False


    def setC3ds(self,datPath,c3ds,fileSuffix=None):

        self.__c3dProcess = True
        self.m_c3dPath = datPath
        self.m_c3ds = c3ds
        self.m_fileSuffix=fileSuffix


    def setThresholdFromOtherAnalysis(self,analysis):
        self.m_thresholdFromAnalysis = analysis

    def setMaxMethod(self,EnumEmgNorm, Value=None):

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
        for c3d in self.m_c3ds:
            filenameOut  = c3d[:-4] + "_" + self.m_fileSuffix+".c3d" if self.m_fileSuffix is not None else c3d
            acq = btkTools.smartReader((self.m_c3dPath+c3d))

            values =  acq.GetAnalog(self.m_label).GetValues()
            valuesNorm = values / self.m_threshold

            btkTools.smartAppendAnalog(acq,self.m_label+"_Norm",valuesNorm, desc= "Normalization)")

            btkTools.smartWriter(acq, (self.m_c3dPath+filenameOut))


    def processAnalysis(self):

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
                logging.warning("[pyCGM2] label [%s] - context [%s] dont find in the emgStats dictionnary"%(self.m_label,contextIt))


    def run(self):
        if self.__c3dProcess:
            self.processC3d()
        self.processAnalysis()




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
