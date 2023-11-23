import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2 import enums
from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Report.normativeDatasets import NormativeData

def pdfGaitReport(DATA_PATH:str,modelledTrials:list[str], 
                  normativeDataset:NormativeData,pointSuffix:str, title:str = "gait report", **kwargs):
    """generate pdf with Kinematic-Kinetic-MAP gait plots .

    Args:
        DATA_PATH (str): data folder path
        modelledTrials (list): modelled trial names
        normativeDataset (pyCGM2.Report.normativeDatasets.NormativeData): a `normativeDataset` instance
        pointSuffix (str): suffix added to model outputs
        title (str,Optional[gait report]): title
    """
    muscleDict=None
    if "musculoSkeletalModel" in kwargs and kwargs["musculoSkeletalModel"]:
        if "modelVersion" in kwargs and kwargs["modelVersion"] == "CGM2.3":
            osimInterface = opensimInterface.osimCgmInterface("CGM2.3")
        if "modelVersion" in kwargs and kwargs["modelVersion"] == "CGM2.2":
            osimInterface = opensimInterface.osimCgmInterface("CGM2.2")
        muscleDict = osimInterface.getMuscles_bySide(addToName="[MuscleLength]")



    analysisInstance =  analysis.makeAnalysis(DATA_PATH,
                            modelledTrials,
                            type="Gait",
                            emgChannels = None,
                            geometryMuscleLabelsDict=muscleDict,
                            dynamicMuscleLabelsDict = None,
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

        if "musculoSkeletalModel" in kwargs and kwargs["musculoSkeletalModel"]:
            # muscle length
            figs,filenames = plot.plot_DescriptiveMuscleLength(DATA_PATH,analysisInstance,
                None,
                normalizedSuffix= None,
                exportPdf=False,
                outputName=title,pointLabelSuffix=pointSuffix,
                show=False,
                title=title)
