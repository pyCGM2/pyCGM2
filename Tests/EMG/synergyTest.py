# -*- coding: utf-8 -*-
import ipdb
import scipy as sp
import matplotlib.pyplot as plt
import logging

import pyCGM2


from pyCGM2.Model.CGM2 import cgm

from pyCGM2.Processing import exporter,c3dManager,cycle,analysis
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import synergy

from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot


class SynergyTest():


    @classmethod
    def analysis(cls):

        # ----DATA-----

        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\analysis\\gaitEMG\\"
        inputFile = ["pre.c3d","post.c3d"]

        EMG_LABELS = ["EMG1","EMG2","EMG3","EMG4","EMG5","EMG6","EMG7","EMG8"]
        EMG_CONTEXT = ["Left","Right","Left","Right","Left","Right","Left","Right"]
        EMG_MUSCLES = ["RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM"]
        NORMAL_ACTIVITIES = ["RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM","RECFEM"]
    #
        for file in inputFile:
            acq = btkTools.smartReader(DATA_PATH+file)
            analysis.processEMG(acq, EMG_LABELS, highPassFrequencies=[20,200],envelopFrequency=6.0)
            btkTools.smartWriter(acq,DATA_PATH+file[:-4]+"-emgProcessed.c3d")

        #inputFileProcessed =   [file[:-4]+"-emgProcessed.c3d" for file in inputFile]

        emgAnalysisPre = analysis.makeEmgAnalysis(DATA_PATH, ["pre-emgProcessed.c3d"], EMG_LABELS)
        emgAnalysisPost = analysis.makeEmgAnalysis(DATA_PATH, ["post-emgProcessed.c3d"], EMG_LABELS)

        analysis.normalizedEMG(emgAnalysisPre, EMG_LABELS,EMG_CONTEXT, method="MeanMax", fromOtherAnalysis=None) # Normalization with the mean of each maximum value across gait cycle
        analysis.normalizedEMG(emgAnalysisPost, EMG_LABELS,EMG_CONTEXT, method="MeanMax", fromOtherAnalysis=emgAnalysisPre)

        #plot.compareEmgEvelops([emgAnalysisPre,emgAnalysisPost],["Pre","Post"], EMG_LABELS, EMG_MUSCLES,EMG_CONTEXT, NORMAL_ACTIVITIES, normalized=True,plotType="Descriptive")

        sfPre = synergy.SynergyFilter(emgAnalysisPre)
        sfPre.getSynergy()

        sfPost = synergy.SynergyFilter(emgAnalysisPost)
        sfPost.getSynergy()

        # tree = sp.spatial.KDTree(sfPre.getW().T)
        # dist,index = tree.query(sfPost.getW().T[0,:])

        #import ipdb; ipdb.set_trace()
        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysisPre)
        exportFilter.export("preAdvancedEMG", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

        exportFilter = exporter.XlsAnalysisExportFilter()
        exportFilter.setAnalysisInstance(emgAnalysisPost)
        exportFilter.export("postAdvancedEMG", path=DATA_PATH,excelFormat = "xls",mode="Advanced")

if __name__ == "__main__":

    plt.close("all")


    SynergyTest.analysis()
