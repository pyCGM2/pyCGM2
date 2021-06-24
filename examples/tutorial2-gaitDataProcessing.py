import pyCGM2
from pyCGM2.Report import normativeDatasets
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot

def example1():

    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\Session 1 - CGM1\\"
    modelledFilenames = ["20180706_CS_PONC_S_NNNN dyn 02.c3d", "20180706_CS_PONC_S_NNNN dyn 03.c3d"]

    analysisInstance = analysis.makeAnalysis(DATA_PATH, modelledFilenames, type="Gait")

    normativeDataset = normativeDatasets.NormativeData("Schwartz2008","Free")

    plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset)
    plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,"LowerLimb",normativeDataset)
    plot.plot_spatioTemporal(DATA_PATH,analysisInstance)

    OUT = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\examples\\"
    analysis.exportAnalysis(analysisInstance,OUT,"spreadsheet")


if __name__ == '__main__':
    example1()
