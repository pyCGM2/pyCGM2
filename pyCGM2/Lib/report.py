# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2 import enums

def pdfGaitReport(DATA_PATH,model,modelledTrials, normativeDataset,pointSuffix, title = "gait report"):

    analysisInstance = analysis.makeAnalysis(
        DATA_PATH,modelledTrials,
        subjectInfo=None,
        experimentalInfo=None,
        modelInfo=None,
        pointLabelSuffix=None)

    title = type

    # spatiotemporal
    plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
        exportPdf=True,
        outputName=title,
        show=None,
        title=title)

    #Kinematics
    if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
        plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)

        plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)
    if model.m_bodypart in [enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
        plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"Trunk",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)

        plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"Trunk",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)

    if model.m_bodypart in [enums.BodyPart.UpperLimb, enums.BodyPart.FullBody]:
        pass # TODO plot upperlimb panel


    #Kinetics
    if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
        plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)

        plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,"LowerLimb",
            normativeDataset,
            exportPdf=True,
            outputName=title,
            pointLabelSuffix=pointSuffix,
            show=False,
            title=title)

    #MAP
    plot.plot_MAP(DATA_PATH,analysisInstance,
        normativeDataset,
        exportPdf=True,
        outputName=title,pointLabelSuffix=pointSuffix,
        show=False,
        title=title)

    plt.show(False)
    logging.info("----- Gait Processing -----> DONE")
