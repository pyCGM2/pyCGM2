# -*- coding: utf-8 -*-
from sklearn.decomposition import NMF
import numpy as np
from pyCGM2.Math import numeric
import matplotlib.pyplot as plt


class SynergyFilter(object):

    def __init__(self,analysisInstance):

        self.m_analysis = analysisInstance

        self.__construct()

    def __construct(self):

        self.m_measured = np.zeros((8,101))
        self.m_measured[0,:] = self.m_analysis.emgStats.data["EMG1_Rectify_Env","Left"]["mean"][:,0]
        self.m_measured[1,:] = self.m_analysis.emgStats.data["EMG2_Rectify_Env","Right"]["mean"][:,0]
        self.m_measured[2,:] = self.m_analysis.emgStats.data["EMG3_Rectify_Env","Left"]["mean"][:,0]
        self.m_measured[3,:] = self.m_analysis.emgStats.data["EMG4_Rectify_Env","Right"]["mean"][:,0]
        self.m_measured[4,:] = self.m_analysis.emgStats.data["EMG5_Rectify_Env","Left"]["mean"][:,0]
        self.m_measured[5,:] = self.m_analysis.emgStats.data["EMG6_Rectify_Env","Right"]["mean"][:,0]
        self.m_measured[6,:] = self.m_analysis.emgStats.data["EMG7_Rectify_Env","Left"]["mean"][:,0]
        self.m_measured[7,:] = self.m_analysis.emgStats.data["EMG8_Rectify_Env","Right"]["mean"][:,0]


    def getSynergy(self, n=6):

        model = NMF(n_components=n, init='random', random_state=0)
        self.m_W = model.fit_transform(self.m_measured)
        self.m_H = model.components_

        self.m_reconstructed = np.dot(self.m_W,self.m_H)

        residu = numeric.rms(self.m_reconstructed-self.m_measured)

    def getW(self):
        return  self.m_W

    def getH(self):
        return  self.m_H
