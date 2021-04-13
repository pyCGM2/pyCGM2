# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

    with PdfPages(DATA_PATH + title+".pdf") as pdf:
        # spatiotemporal
        plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
            exportPdf=False,
            outputName=title,
            show=None,
            title=title)
        pdf.savefig()

        #Kinematics
        if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
            plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

            plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"LowerLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

        if model.m_bodypart in [enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
            plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"Trunk",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

            plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"Trunk",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

        if model.m_bodypart in [enums.BodyPart.UpperLimb, enums.BodyPart.FullBody]:
            plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"UpperLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

            plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,"UpperLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()


        #Kinetics
        if model.m_bodypart in [enums.BodyPart.LowerLimb,enums.BodyPart.LowerLimbTrunk, enums.BodyPart.FullBody]:
            plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

            plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,"LowerLimb",
                normativeDataset,
                exportPdf=False,
                outputName=title,
                pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
            pdf.savefig()

        #MAP
        plot.plot_MAP(DATA_PATH,analysisInstance,
            normativeDataset,
            exportPdf=False,
            outputName=title,pointLabelSuffix=pointSuffix,
            show=False,
            title=title)
        pdf.savefig()


        LOGGER.logger.info("----- Gait Processing -----> DONE")
