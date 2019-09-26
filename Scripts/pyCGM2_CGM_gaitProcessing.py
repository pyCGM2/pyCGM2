# -*- coding: utf-8 -*-
import os
import argparse
import traceback
import logging

import pyCGM2
from pyCGM2.Utils import files
from pyCGM2.Configurator import ProcessingManager
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Processing import exporter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from pyCGM2 import log;
log.setLogger()

def main(args):
    DATA_PATH = os.getcwd()+"\\"
    userSettings = files.openFile(DATA_PATH,args.userFile)

    manager = ProcessingManager.GaitProcessingConfigManager(userSettings,modelVersion=None)

    modelledTrials = [str(trialFilename[:-4]+"-pyCGM2modelled.c3d") for trialFilename in manager.modelledTrials]

    analysisInstance = analysis.makeAnalysis(
            DATA_PATH,modelledTrials,
            subjectInfo=manager.subjectInfo,
            experimentalInfo=manager.experimentalInfo,
            modelInfo=None,
            pointLabelSuffix=manager.pointSuffix)

    plot.plot_spatioTemporal(DATA_PATH,analysisInstance,
        exportPdf=True,
        outputName=manager.title)

    for body in manager.bodyPart:
        if not manager.consistencyFlag:
            plot.plot_DescriptiveKinematic(DATA_PATH,analysisInstance,body,
                manager.normativeData,
                exportPdf=True,
                outputName=manager.title,
                pointLabelSuffix=manager.pointSuffix)
            try:
                plot.plot_DescriptiveKinetic(DATA_PATH,analysisInstance,body,
                    manager.normativeData,
                    exportPdf=True,outputName=manager.title,
                    pointLabelSuffix=manager.pointSuffix)
            except:
                pass

        else:

            plot.plot_ConsistencyKinematic(DATA_PATH,analysisInstance,body,
                manager.normativeData,
                exportPdf=True,
                outputName=manager.title,
                pointLabelSuffix=manager.pointSuffix)

            try:
                plot.plot_ConsistencyKinetic(DATA_PATH,analysisInstance,body,
                    manager.normativeData,
                    exportPdf=True,outputName=manager.title,
                    pointLabelSuffix=manager.pointSuffix)
            except:
                pass

    plot.plot_MAP(DATA_PATH,analysisInstance,manager.normativeData,exportPdf=True,outputName=manager.title,pointLabelSuffix=manager.pointSuffix)

    exportXlsFilter = exporter.XlsAnalysisExportFilter()
    exportXlsFilter.setAnalysisInstance(analysisInstance)
    exportXlsFilter.export(manager.title, path=DATA_PATH,excelFormat = "xls",mode="Advanced")

    exportC3dFilter = exporter.AnalysisC3dExportFilter()
    exportC3dFilter.setAnalysisInstance(analysisInstance)
    exportC3dFilter.export(manager.title, path=DATA_PATH)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='CGM-processing')
    parser.add_argument('--userFile', type=str, help='userSettings', default="GaitProcessing.userSettings")

    args = parser.parse_args()
        #print args

        # ---- main script -----
    try:
        main(args)


    except Exception, errormsg:
        print "Script errored!"
        print "Error message: %s" % errormsg
        traceback.print_exc()
        print "Press return to exit.."
        #raw_input()
        #
