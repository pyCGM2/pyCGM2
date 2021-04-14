# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2 import enums

def pdfGaitReport(DATA_PATH,model,modelledTrials, normativeDataset,pointSuffix, title = "gait report"):

    analysisInstance =  analysis.makeAnalysis(DATA_PATH,
                            modelledTrials,
                            type="Gait",
                            emgChannels = None,
                            pointLabelSuffix=None,
                            subjectInfo=None, experimentalInfo=None,modelInfo=None,
                            )


    analysis.exportAnalysis(analysisInstance,DATA_PATH,title, mode="Advanced")

    with PdfPages(DATA_PATH + title+".pdf") as pdf:
        # spatiotemporal
        plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
            exportPdf=False,
            outputName=title,
            show=None,
            title=title)
        pdf.savefig()

        #Kinematics
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
