import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Report import plot,plotFilters,plotViewers,normativeDatasets,annotator
from pyCGM2.Processing import c3dManager,exporter,scores
from pyCGM2.Processing.highLevel import gaitSmartFunctions
from pyCGM2.Utils import files
from pyCGM2.Processing import jointPatterns


DATA_PATH = "C:/Users/HLS501/Documents/VICON DATA/pyCGM2-Data/Datasets Tests/didier/08_02_18_Vincent Pere/"

modelledFiles = ["08_02_18_Vincent_Pere_Gait_000_MOKKA-modelled-cgm24.c3d"]
modelInfo = None
subjectInfo = None
experimentalInfo = None
modelVersion= "CGM2.4"
pointSuffix=""


# analysis constructor-------
c3dmanagerProcedure = c3dManager.UniqueC3dSetProcedure(DATA_PATH,modelledFiles)
cmf = c3dManager.C3dManagerFilter(c3dmanagerProcedure)
cmf.enableEmg(False)
trialManager = cmf.generate()

analysis = gaitSmartFunctions.make_analysis(trialManager,
      None,
      None,
      modelInfo, subjectInfo, experimentalInfo,
      modelVersion = "CGM2.4",
      pointLabelSuffix=pointSuffix)
# end analysis -------
normativeDataset = normativeDatasets.Schwartz2008("Free")

kv = plotViewers.LowerLimbMultiFootKinematicsPlotViewer(analysis,
                    pointLabelSuffix=pointSuffix)

kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
kv.setNormativeDataset(normativeDataset)

# filter
pf = plotFilters.PlottingFilter()
pf.setViewer(kv)
pf.plot()

plt.show()
